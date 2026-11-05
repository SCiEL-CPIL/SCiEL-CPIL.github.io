"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize, minimize_scalar
import sys

N = 400

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _eval_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _soft_max_grad(x, beta):
    """Soft maximum and its gradient."""
    m = np.max(x)
    if not np.isfinite(m):
        return m, np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    w = exp_vals / np.sum(exp_vals)
    sm = m + np.log(np.sum(exp_vals)) / beta
    grad = beta * w
    return sm, grad

def _chebyshev_basis(k, x):
    """Chebyshev polynomial of second kind U_k(x)."""
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return 2.0 * x
    else:
        u_prev = np.ones_like(x)
        u_curr = 2.0 * x
        for _ in range(2, k + 1):
            u_next = 2.0 * x * u_curr - u_prev
            u_prev = u_curr
            u_curr = u_next
        return u_curr

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Chebyshev polynomial parameterization for structured zero-mean perturbations,
    combined with DCT-domain optimization and soft-max surrogate.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Uniform grid on [-1, 1]
    x_grid = np.linspace(-1.0, 1.0, N)

    # Phase 1: Chebyshev polynomial families
    # h[i] = 1 + sum_{k=1}^{M} a_k * U_k(x_i)
    # U_k are orthogonal on [-1,1] with weight sqrt(1-x^2)
    # For uniform grid, they're approximately orthogonal
    
    # Try single Chebyshev modes
    for M in range(1, 30):
        if time.time() > deadline - 100:
            break
        
        # Compute U_M on the grid
        uM = _chebyshev_basis(M, x_grid)
        sum_uM = np.sum(uM)
        
        if abs(sum_uM) < 1e-10:
            # Already zero mean, just scale
            for scale in np.linspace(-2.0, 2.0, 41):
                h = 1.0 + scale * uM
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
        else:
            # Need to adjust to maintain sum = N
            # h = 1 + scale * (uM - mean(uM))
            uM_centered = uM - np.mean(uM)
            for scale in np.linspace(-2.0, 2.0, 41):
                h = 1.0 + scale * uM_centered
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 2: Two-mode Chebyshev combinations
    if time.time() < deadline - 80:
        for M1 in range(1, 15):
            for M2 in range(M1 + 1, min(M1 + 5, 25)):
                if time.time() > deadline - 80:
                    break
                
                u1 = _chebyshev_basis(M1, x_grid) - np.mean(_chebyshev_basis(M1, x_grid))
                u2 = _chebyshev_basis(M2, x_grid) - np.mean(_chebyshev_basis(M2, x_grid))
                
                for a1 in np.linspace(-1.5, 1.5, 11):
                    for a2 in np.linspace(-1.5, 1.5, 11):
                        h = 1.0 + a1 * u1 + a2 * u2
                        v = _eval_value(h)
                        if v < best_val:
                            best_val = v
                            best_h = h.copy()

    # Phase 3: DCT-domain optimization with soft-max surrogate
    # Convert best_h to DCT coefficients
    c_best = dct(best_h, type=2, norm='ortho')
    c_dc = c_best[0].copy()

    def _dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')

    # Multi-stage soft-max optimization in DCT domain
    betas = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break

        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)

            if not np.all(np.isfinite(h_recon)):
                return 1e10

            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10

            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        # Multiple random restarts
        rng = np.random.RandomState(42 + int(beta) % 100)
        c_ac_init = c_best[1:].copy()

        n_restarts = 3 if beta < 50 else 2
        for trial in range(n_restarts):
            if time.time() > deadline - 20:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                # Perturb with decreasing amplitude for higher beta
                amp = 0.2 / (1 + trial)
                c_ac_trial = c_ac_init + amp * rng.randn(N - 1)

            try:
                max_iter = 300 if beta < 100 else 200
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = dct(best_h, type=2, norm='ortho')
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 4: High-precision polish with very high beta
    if time.time() < deadline - 10:
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() > deadline - 5:
                break

            def obj_polish(c_ac, beta=beta):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = _dct_to_h(c_full)

                if not np.all(np.isfinite(h_recon)):
                    return 1e10

                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10

                h_norm = h_recon * (N / s)
                conv = _convolve_fft(h_norm)
                sm = _soft_max(conv, beta)
                return abs(2.0 * N * sm / (N * N))

            try:
                result = minimize(
                    obj_polish,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = dct(best_h, type=2, norm='ortho')
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # Phase 5: Final exact polish - try to directly minimize the max
    if time.time() < deadline - 3:
        # Use the current best and try a few more DCT optimizations
        for beta in [10000.0]:
            if time.time() > deadline - 2:
                break

            def obj_final(c_ac, beta=beta):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = _dct_to_h(c_full)
                if not np.all(np.isfinite(h_recon)):
                    return 1e10
                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10
                h_norm = h_recon * (N / s)
                conv = _convolve_fft(h_norm)
                sm = _soft_max(conv, beta)
                return abs(2.0 * N * sm / (N * N))

            try:
                result = minimize(
                    obj_final,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
