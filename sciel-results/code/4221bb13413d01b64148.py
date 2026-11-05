"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses ADMM-style epigraph optimization with auxiliary variable t for the
    autoconvolution peak, combined with spectral (FFT) updates and DCT-domain
    refinement.
    """
    start = time.time()
    deadline = start + 142.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric exactly."""
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (h_sum ** 2))

    def autoconv_fft(h):
        """Compute linear autoconvolution efficiently via FFT."""
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        H = fft(h, n=n_fft)
        conv = ifft(H * H, n=n_fft).real[:n_conv]
        return conv

    def soft_max(x, beta):
        """Numerically stable soft maximum."""
        m = np.max(x)
        if not np.isfinite(m):
            return m
        exp_vals = np.exp(beta * (x - m))
        sm = m + np.log(np.sum(exp_vals)) / beta
        return sm

    # --- Phase 1: Parametric Initialization ---
    # Try simple families to get a good starting point better than ones.
    
    # Family 1: Cosine waves
    for m in range(1, 21):
        if time.time() > deadline - 85:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = compute_value(h)
            if v < best_value:
                best_value = v
                best_h = h.copy()

    # Family 2: Smoothed block (tanh)
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 85:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = compute_value(h)
        if v < best_value:
            best_value = v
            best_h = h.copy()

    # Family 3: Sine waves (odd symmetry can reduce peak)
    for m in range(1, 15):
        if time.time() > deadline - 85:
            break
        for a in np.linspace(-0.8, 0.8, 13):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * np.arange(N) / N)
            v = compute_value(h)
            if v < best_value:
                best_value = v
                best_h = h.copy()

    # --- Phase 2: ADMM Epigraph Optimization ---
    # Minimize t subject to: conv(h) <= t (element-wise), sum(h) = N
    # Using ADMM with splitting: h-update via FFT, t-update via projection
    
    def admm_optimize(h_init, max_iters=500, rho=10.0, tol=1e-8, time_budget=None):
        """
        ADMM for: min t s.t. conv(h) <= t, sum(h) = N
        Variables: h (heights), t (scalar peak bound), u (dual for conv constraint)
        """
        h = h_init.copy().astype(float)
        # Normalize to sum=N
        s = np.sum(h)
        if abs(s) < 1e-10:
            return h_init, np.inf
        h = h * (N / s)
        
        t = np.max(autoconv_fft(h))
        u = np.zeros(2 * N - 1)  # dual variable for conv <= t
        
        best_local_h = h.copy()
        best_local_t = t
        best_local_val = compute_value(h)
        
        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            
            # --- h-update: minimize 0.5*||conv(h) - t + u||^2 + (rho/2)*||h - h_old||^2
            #     subject to sum(h) = N
            # In frequency domain: minimize ||F{h}^2 - (t - u)||^2 + (rho/2)*||F{h} - F{h_old}||^2
            # This is still non-trivial; use a proximal gradient step in h-space
            
            # Compute current conv
            conv_h = autoconv_fft(h)
            residual = conv_h - t + u
            
            # Gradient of 0.5*||conv(h) - t + u||^2 w.r.t. h
            # conv(h) = h * h (linear), so d(conv)/dh is a Toeplitz matrix
            # For efficiency, compute gradient via FFT:
            # grad_h = 2 * (conv(h - t + u) * h) in appropriate form
            # Actually: d/dh [sum_k (conv(h)_k - t_k + u_k)^2]
            # conv(h)_k = sum_i h_i h_{k-i}
            # d/dh_j conv(h)_k = h_{k-j} + h_{k-j} (with boundary handling)
            # Use FFT-based approach:
            # Let r = conv(h) - t + u (length 2N-1)
            # grad_h[j] = sum_k r_k * (d conv_k / d h_j)
            # = sum_k r_k * (h_{k-j} + h_{k-j}) with proper indexing
            # Simpler: use the fact that in frequency domain,
            # conv(h) = ifft(fft(h)^2), so we can do:
            
            # Use subgradient step: h_new = h - alpha * grad
            # Compute gradient efficiently:
            # grad_h = 2 * ifft(fft(h) * fft(r_padded)) where r_padded is r embedded
            # Actually: conv(h) = ifft(|H|^2) where H = fft(h)
            # d/dh sum_k (conv_k - t_k + u_k)^2 
            # = 2 * ifft( H * fft(r) ) where r is padded to FFT length
            
            n_conv = 2 * N - 1
            n_fft = 1
            while n_fft < n_conv:
                n_fft *= 2
            
            H = fft(h, n=n_fft)
            r = conv_h - t + u  # length n_conv
            r_padded = np.zeros(n_fft)
            r_padded[:n_conv] = r
            R = fft(r_padded)
            
            # Gradient: 2 * ifft(H * R) truncated to N
            grad_h = 2.0 * ifft(H * R, n=n_fft).real[:N]
            
            # Add constraint gradient for sum(h) = N: lambda * ones
            # Use augmented Lagrangian: add rho*(sum(h) - N) * ones
            sum_h = np.sum(h)
            lambda_sum = rho * (sum_h - N)
            grad_h += lambda_sum
            
            # Step size
            alpha = 0.5 / rho
            h_new = h - alpha * grad_h
            
            # Project to sum(h) = N
            sum_new = np.sum(h_new)
            h_new = h_new + (N - sum_new) / N
            
            # --- t-update: minimize t + (1/rho)*||t - (conv(h_new) - u)||^2
            # t_new = max(conv(h_new) - u)
            conv_new = autoconv_fft(h_new)
            t_new = np.max(conv_new - u)
            
            # --- u-update: u += conv(h_new) - t_new
            u = u + conv_new - t_new
            
            # Update
            h = h_new
            t = t_new
            
            # Check convergence
            primal_res = np.max(np.abs(conv_new - t))
            if primal_res < tol:
                break
            
            # Track best
            val = compute_value(h)
            if val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
        
        return best_local_h, best_local_val

    if time.time() < deadline - 50:
        # Run ADMM from current best
        h_admm, val_admm = admm_optimize(
            best_h, max_iters=300, rho=5.0, 
            time_budget=deadline - 45
        )
        if val_admm < best_value:
            best_value = val_admm
            best_h = h_admm.copy()
        
        # Try different rho values
        for rho_val in [1.0, 20.0, 50.0]:
            if time.time() > deadline - 40:
                break
            h_admm, val_admm = admm_optimize(
                best_h, max_iters=200, rho=rho_val,
                time_budget=deadline - 35
            )
            if val_admm < best_value:
                best_value = val_admm
                best_h = h_admm.copy()

    # --- Phase 3: DCT Domain Optimization with Soft-Max ---
    def h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')

    c_best = h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Multi-stage beta annealing in DCT domain
    betas = [5.0, 20.0, 50.0, 100.0, 300.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break

        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = dct_to_h(c_full)

            if not np.all(np.isfinite(h_recon)):
                return 1e10

            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10

            h_norm = h_recon * (N / s)
            conv = autoconv_fft(h_norm)
            sm = soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        # Try a few restarts
        rng = np.random.RandomState(42 + int(beta))
        
        for trial in range(3):
            if time.time() > deadline - 18:
                break

            if trial == 0:
                c_ac_trial = c_best[1:].copy()
            else:
                c_ac_trial = c_best[1:].copy() + 0.05 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = compute_value(h_opt)
                    if v_opt < best_value:
                        best_value = v_opt
                        best_h = h_opt.copy()
                        c_best = h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # --- Phase 4: Final ADMM Polish ---
    if time.time() < deadline - 10:
        h_final, val_final = admm_optimize(
            best_h, max_iters=400, rho=10.0,
            time_budget=deadline - 5
        )
        if val_final < best_value:
            best_value = val_final
            best_h = h_final.copy()

    return best_h
# EVOLVE-BLOCK-END
