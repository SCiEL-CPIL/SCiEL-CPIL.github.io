"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

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

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    h is normalized internally so sum(h) = N.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chebyshev_basis(M, x):
    """
    Compute Chebyshev polynomials of the second kind U_0(x), ..., U_{M-1}(x)
    evaluated at points x. Returns array of shape (M, len(x)).
    Uses recurrence: U_0=1, U_1=2x, U_k=2x*U_{k-1} - U_{k-2}
    """
    n = len(x)
    basis = np.zeros((M, n))
    if M > 0:
        basis[0] = 1.0
    if M > 1:
        basis[1] = 2.0 * x
    for k in range(2, M):
        basis[k] = 2.0 * x * basis[k-1] - basis[k-2]
    return basis

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses multiple optimization strategies:
    1. Parametric families for good initial points
    2. Random walk with drift
    3. DCT-domain optimization with soft-max surrogate
    4. Chebyshev polynomial parameterization
    5. High-precision polish
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N)
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Random walk with negative drift
    if time.time() < deadline - 60:
        rng = np.random.RandomState(42)
        
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if time.time() > deadline - 60:
                break
            
            for trial in range(20):
                if time.time() > deadline - 60:
                    break
                
                noise = rng.randn(N) * noise_scale
                walk = np.cumsum(noise)
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                h = 1.0 + walk - drift * np.arange(N)
                
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Phase 3: DCT-domain optimization with soft-max surrogate
    from scipy.fft import dct, idct
    
    def _h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    for beta in [1.0, 5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 25:
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

        rng = np.random.RandomState(42 + int(beta))
        c_ac_init = c_best[1:].copy()

        for trial in range(4):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_ac_init.copy()
            else:
                c_ac_trial = c_ac_init + 0.1 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
                        c_ac_init = c_best[1:].copy()
            except Exception:
                continue

    # Phase 4: Chebyshev polynomial parameterization
    # h[i] = sum_{k=0}^{M-1} a_k * U_k(x_i) where x_i = 2*i/(N-1) - 1
    # This exploits the bounded autoconvolution properties of orthogonal polynomials
    if time.time() < deadline - 20:
        x_grid = 2.0 * np.arange(N) / (N - 1) - 1.0  # uniform grid on [-1, 1]
        
        for M in [4, 6, 8, 10, 12]:
            if time.time() > deadline - 15:
                break
            
            # Precompute Chebyshev basis
            basis = _chebyshev_basis(M, x_grid)
            
            # Initialize coefficients: start from projection of best_h onto Chebyshev basis
            # Use least squares to project
            try:
                a_init, _, _, _ = np.linalg.lstsq(basis.T, best_h, rcond=None)
            except:
                a_init = np.zeros(M)
                a_init[0] = np.mean(best_h)
            
            # Optimize coefficients with soft-max surrogate
            for beta in [5.0, 20.0, 50.0, 100.0]:
                if time.time() > deadline - 12:
                    break
                
                def obj_cheb(a, beta=beta, basis=basis):
                    h_recon = basis.T @ a
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
                        obj_cheb,
                        a_init.copy(),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    a_opt = result.x
                    h_opt = basis.T @ a_opt
                    
                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _objective_exact(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                            # Update DCT representation for subsequent phases
                            c_best = _h_to_dct(best_h)
                            c_dc = c_best[0].copy()
                except Exception:
                    continue

    # Phase 5: High-precision polish with very high beta in DCT domain
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
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
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # Phase 6: Final high-precision polish with epigraph-style approach
    # Use auxiliary variable t to handle max more smoothly
    if time.time() < deadline - 5:
        # Try a few more DCT optimizations with very tight tolerances
        for beta in [2000.0, 5000.0]:
            if time.time() > deadline - 3:
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
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
