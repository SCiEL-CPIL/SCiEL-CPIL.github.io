"""Combined program: ADMM epigraph + spectral flattening + DCT refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines ADMM epigraph optimization with spectral flattening (L4 power penalty)
    and DCT-domain refinement. The spectral flattening phase alternates with peak
    minimization to escape local minima by trading off between flatness and peak height.
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

    def power_spectrum_l4(h):
        """Compute L4 norm of power spectrum (sum of |H|^8), encouraging flatness."""
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        H = fft(h, n=n_fft)
        power = np.abs(H) ** 2
        # L4 norm of power spectrum: sum(power^4) = sum(|H|^8)
        return np.sum(power ** 4)

    # --- Phase 1: Parametric Initialization ---
    for m in range(1, 21):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = compute_value(h)
            if v < best_value:
                best_value = v
                best_h = h.copy()

    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 90:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = compute_value(h)
        if v < best_value:
            best_value = v
            best_h = h.copy()

    for m in range(1, 15):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-0.8, 0.8, 13):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * np.arange(N) / N)
            v = compute_value(h)
            if v < best_value:
                best_value = v
                best_h = h.copy()

    # --- Phase 2: ADMM Epigraph Optimization ---
    def admm_optimize(h_init, max_iters=500, rho=10.0, tol=1e-8, time_budget=None):
        """
        ADMM for: min t s.t. conv(h) <= t (element-wise), sum(h) = N
        """
        h = h_init.copy().astype(float)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return h_init, np.inf
        h = h * (N / s)
        
        t = np.max(autoconv_fft(h))
        u = np.zeros(2 * N - 1)
        
        best_local_h = h.copy()
        best_local_t = t
        best_local_val = compute_value(h)
        
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        
        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            
            conv_h = autoconv_fft(h)
            residual = conv_h - t + u
            
            H = fft(h, n=n_fft)
            r = conv_h - t + u
            r_padded = np.zeros(n_fft)
            r_padded[:n_conv] = r
            R = fft(r_padded)
            
            grad_h = 2.0 * ifft(H * R, n=n_fft).real[:N]
            
            sum_h = np.sum(h)
            lambda_sum = rho * (sum_h - N)
            grad_h += lambda_sum
            
            alpha = 0.5 / rho
            h_new = h - alpha * grad_h
            
            sum_new = np.sum(h_new)
            h_new = h_new + (N - sum_new) / N
            
            conv_new = autoconv_fft(h_new)
            t_new = np.max(conv_new - u)
            
            u = u + conv_new - t_new
            
            h = h_new
            t = t_new
            
            primal_res = np.max(np.abs(conv_new - t))
            if primal_res < tol:
                break
            
            val = compute_value(h)
            if val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
        
        return best_local_h, best_local_val

    if time.time() < deadline - 60:
        h_admm, val_admm = admm_optimize(
            best_h, max_iters=300, rho=5.0, 
            time_budget=deadline - 55
        )
        if val_admm < best_value:
            best_value = val_admm
            best_h = h_admm.copy()
        
        for rho_val in [1.0, 20.0, 50.0]:
            if time.time() > deadline - 50:
                break
            h_admm, val_admm = admm_optimize(
                best_h, max_iters=200, rho=rho_val,
                time_budget=deadline - 45
            )
            if val_admm < best_value:
                best_value = val_admm
                best_h = h_admm.copy()

    # --- Phase 3: Spectral Flattening via Alternating Optimization ---
    # Alternate between minimizing soft-max peak and L4 power spectrum penalty
    # This encourages flat power spectrum, reducing peak-to-average ratio
    
    def combined_objective(h, beta_sm, lambda_spec):
        """Combined objective: soft-max peak + lambda * L4 power spectrum penalty."""
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        h_norm = h * (N / s)
        conv = autoconv_fft(h_norm)
        sm = soft_max(conv, beta_sm)
        spec_penalty = power_spectrum_l4(h_norm)
        obj = sm + lambda_spec * spec_penalty
        return obj, h_norm

    def spectral_flattening_optimize(h_init, max_iters=100, beta_sm=50.0, 
                                       lambda_spec=0.01, time_budget=None):
        """
        Alternating optimization: minimize combined objective via projected gradient.
        """
        h = h_init.copy().astype(float)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return h_init, np.inf
        h = h * (N / s)
        
        best_local_h = h.copy()
        best_local_val = compute_value(h)
        
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        
        # Adaptive learning rate
        lr = 0.01
        patience = 0
        max_patience = 20
        
        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            
            # Compute gradient of combined objective
            conv_h = autoconv_fft(h)
            
            # Gradient of soft-max
            m = np.max(conv_h)
            exp_vals = np.exp(beta_sm * (conv_h - m))
            weights = exp_vals / np.sum(exp_vals)
            
            H = fft(h, n=n_fft)
            w_padded = np.zeros(n_fft)
            w_padded[:n_conv] = weights
            W = fft(w_padded)
            
            grad_sm = 2.0 * ifft(H * W, n=n_fft).real[:N]
            
            # Gradient of L4 power spectrum penalty
            # d/dh sum(|H|^8) = 8 * Re(ifft(H^3 * conj(H) * conj(H) * conj(H) * conj(H)))
            # More simply: d/dh sum(|H|^(2k)) = 2k * Re(ifft(H^(k-1) * conj(H)^k))
            # For k=4: d/dh sum(|H|^8) = 8 * Re(ifft(H^3 * conj(H)^4))
            # = 8 * Re(ifft(H^3 * conj(H)^4))
            # Let's compute: H^3 * conj(H)^4 = H^3 * conj(H)^3 * conj(H) = |H|^6 * conj(H)
            # So gradient = 8 * Re(ifft(|H|^6 * conj(H)))
            power6 = np.abs(H) ** 6
            grad_spec = 8.0 * ifft(power6 * np.conj(H), n=n_fft).real[:N]
            
            # Total gradient
            grad = grad_sm + lambda_spec * grad_spec
            
            # Normalize gradient for stable steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Projected gradient step with sum constraint
            h_new = h - lr * grad
            sum_new = np.sum(h_new)
            h_new = h_new + (N - sum_new) / N
            
            # Evaluate
            val = compute_value(h_new)
            if val < best_local_val:
                best_local_val = val
                best_local_h = h_new.copy()
                h = h_new.copy()
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    # Reduce learning rate and reset patience
                    lr *= 0.5
                    patience = 0
                    if lr < 1e-6:
                        break
            
            # Increase lambda_spec gradually to emphasize flattening
            lambda_spec *= 1.05
        
        return best_local_h, best_local_val

    if time.time() < deadline - 40:
        # Run spectral flattening from current best
        h_spec, val_spec = spectral_flattening_optimize(
            best_h, max_iters=80, beta_sm=50.0, lambda_spec=0.005,
            time_budget=deadline - 35
        )
        if val_spec < best_value:
            best_value = val_spec
            best_h = h_spec.copy()
        
        # Try different lambda values
        for lam in [0.001, 0.01, 0.05, 0.1]:
            if time.time() > deadline - 30:
                break
            h_spec, val_spec = spectral_flattening_optimize(
                best_h, max_iters=60, beta_sm=50.0, lambda_spec=lam,
                time_budget=deadline - 25
            )
            if val_spec < best_value:
                best_value = val_spec
                best_h = h_spec.copy()

    # --- Phase 4: DCT Domain Optimization with Combined Objective ---
    def h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')

    c_best = h_to_dct(best_h)
    c_dc = c_best[0].copy()

    betas = [5.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break

        def obj_dct(c_ac, beta=beta, lambda_spec=0.01):
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
            spec_pen = power_spectrum_l4(h_norm)
            return sm + lambda_spec * spec_pen

        rng = np.random.RandomState(42 + int(beta))
        
        for trial in range(3):
            if time.time() > deadline - 12:
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
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
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

    # --- Phase 5: Final ADMM Polish ---
    if time.time() < deadline - 10:
        h_final, val_final = admm_optimize(
            best_h, max_iters=300, rho=10.0,
            time_budget=deadline - 5
        )
        if val_final < best_value:
            best_value = val_final
            best_h = h_final.copy()

    # Ensure the returned h is valid
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_value = 2.0

    return best_h
# EVOLVE-BLOCK-END
