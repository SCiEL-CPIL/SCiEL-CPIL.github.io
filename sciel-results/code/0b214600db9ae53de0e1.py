import numpy as np
import time

N = 600
W = 1.0 / 1200.0

# EVOLVE-BLOCK-START
def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT on a fine grid for accurate peak detection.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    n_grid = 2048
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _fourier_to_heights(coefs, N, n_modes):
    """Convert Fourier coefficients to step heights.
    coefs[0] = DC (mean), coefs[1:2] = mode 1 (cos, sin), etc.
    Returns nonnegative heights."""
    i = np.arange(N)
    t = i / N
    h = np.zeros(N)
    h += coefs[0]
    for k in range(1, n_modes + 1):
        c = coefs[2*k - 1]
        s = coefs[2*k]
        h += c * np.cos(2 * np.pi * k * t) + s * np.sin(2 * np.pi * k * t)
    return np.maximum(h, 0.0)

def _heights_to_fourier(h, N, n_modes):
    """Convert step heights to Fourier coefficients (least-squares projection)."""
    i = np.arange(N)
    t = i / N
    coefs = np.zeros(2 * n_modes + 1)
    coefs[0] = np.mean(h)
    for k in range(1, n_modes + 1):
        cos_k = np.cos(2 * np.pi * k * t)
        sin_k = np.sin(2 * np.pi * k * t)
        coefs[2*k - 1] = np.mean(h * cos_k) * 2
        coefs[2*k] = np.mean(h * sin_k) * 2
    return coefs

def _fourier_gradient_descent(h_init, deadline, n_modes=20, max_iter=2000, beta=50.0):
    """
    Optimize in Fourier space using Gradient Descent with Log-Sum-Exp approximation for max.
    """
    coefs = _heights_to_fourier(h_init, N, n_modes)
    best_coefs = coefs.copy()
    best_ratio = _compute_ratio_fast(_fourier_to_heights(coefs, N, n_modes))
    
    # Precompute Fourier basis matrix A (N x (2*n_modes+1))
    i = np.arange(N)
    t = i / N
    n_coefs = 2 * n_modes + 1
    A = np.zeros((N, n_coefs))
    A[:, 0] = 1.0
    for k in range(1, n_modes + 1):
        A[:, 2*k - 1] = np.cos(2 * np.pi * k * t)
        A[:, 2*k] = np.sin(2 * np.pi * k * t)
    
    lr = 0.01
    lr_decay = 0.99
    momentum = 0.9
    v = np.zeros(n_coefs)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        h = A @ coefs
        h = np.maximum(h, 0.0)
        
        # Compute autoconv on fine grid for gradient
        n_grid = 2048
        x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
        cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
        f_fine = h[cell_idx]
        
        n_fft = 2 * n_grid
        F = np.fft.rfft(f_fine, n_fft)
        conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
        dx = 0.5 / n_grid
        conv *= dx
        
        # Log-Sum-Exp for max
        # Scale to avoid overflow
        max_c = np.max(conv)
        if max_c < 1e-15:
            break
        exp_conv = np.exp(beta * (conv - max_c))
        sum_exp = np.sum(exp_conv)
        P = (max_c + np.log(sum_exp) / beta) # Soft max value
        
        # Weights for gradient of P
        w = exp_conv / sum_exp
        
        # Gradient of P w.r.t h
        # dP/dh_i = sum_m w_m * d(conv_m)/dh_i
        # conv_m = sum_k f_k f_{m-k}
        # This is effectively correlation of w and f_fine
        # grad_h_fine = 2 * (w * f_fine) (correlation)
        # Use FFT for correlation
        W_fft = np.fft.rfft(w, n_fft)
        F_fft = np.fft.rfft(f_fine, n_fft)
        corr = np.fft.irfft(W_fft * F_fft, n_fft)
        # Take the first n_grid elements corresponding to the linear convolution part?
        # Actually, autoconv of length N on grid N_grid.
        # The gradient w.r.t f_fine is 2 * correlation(w, f_fine)
        # We need to map this back to h (N bins)
        
        # Simpler: Compute gradient w.r.t f_fine, then sum over fine grid points to get gradient w.r.t h
        grad_f_fine = 2.0 * corr[:n_grid] # Approximation, need to handle boundaries carefully
        # Actually, the correlation `corr` computed via FFT of zero-padded signals gives the full linear correlation.
        # The term `conv` was `irfft(F*F)[:2*n_grid-1]`.
        # The gradient of `sum w_m conv_m` w.r.t `f_fine` is `2 * (w * f_fine)` where `*` is correlation.
        
        # Map grad_f_fine to grad_h
        grad_h = np.zeros(N)
        # For each fine grid point, it belongs to a bin.
        # grad_h[bin] += grad_f_fine[point] * (dx / bin_width)? 
        # h is constant on bins. f_fine is h[bin].
        # d(f_fine)/d(h[bin]) = 1 for points in bin.
        # So grad_h[bin] = sum_{p in bin} grad_f_fine[p] * dx ? No, h is the value.
        # The energy is sum h^2.
        # Let's just use the chain rule:
        # P depends on f_fine. f_fine depends on h.
        # grad_h = sum_{p} (d f_fine_p / d h) grad_f_fine_p
        # f_fine_p = h[cell_idx[p]].
        # So grad_h[cell_idx[p]] += grad_f_fine[p]
        
        np.add.at(grad_h, cell_idx, grad_f_fine)
        
        # Gradient of L = sum f_fine^2 * dx
        # L = sum h^2 * (n_grid/N) * dx ?
        # f_fine has N_grid points. h has N points.
        # L2_sq = sum(f_fine^2) * dx.
        # f_fine = h[cell].
        # dL/dh[bin] = 2 * sum_{p in bin} f_fine[p] * dx = 2 * h[bin] * (n_grid/N) * dx
        # Note: sum_{p in bin} 1 = n_grid/N (approx).
        # So dL/dh = 2 * h * (n_grid/N) * dx
        
        l2_sq = np.sum(f_fine ** 2) * dx
        dL_dh = 2.0 * h * (n_grid / N) * dx
        
        # Gradient of Ratio R = P / L
        # dR = (dP * L - P * dL) / L^2
        if l2_sq < 1e-15:
            break
            
        grad_R_h = (grad_h * l2_sq - P * dL_dh) / (l2_sq ** 2)
        
        # Map to Fourier coefficients
        # h = A coefs
        # grad_coefs = A^T grad_R_h
        grad_coefs = A.T @ grad_R_h
        
        # Update with momentum
        v = momentum * v - lr * grad_coefs
        coefs += v
        
        # Keep mean non-negative
        coefs[0] = max(0.0, coefs[0])
        
        # Evaluate true ratio (max, not soft max)
        h_curr = _fourier_to_heights(coefs, N, n_modes)
        r_curr = _compute_ratio_fast(h_curr)
        
        if r_curr < best_ratio:
            best_ratio = r_curr
            best_coefs = coefs.copy()
            
        lr *= lr_decay
        
    return _fourier_to_heights(best_coefs, N, n_modes), best_ratio

def _coordinate_descent(h, deadline, max_iter=200):
    """Final polish: coordinate descent on raw heights."""
    best_ratio = _compute_ratio_fast(h)
    h = h.copy()
    step_size = max(0.005, np.max(h) * 0.01)
    rng = np.random.RandomState(7)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        n_trials = 40
        idxs = rng.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            
            # Ensure non-negativity for minus step
            new_val_minus = max(0.0, h_old - step_size)
            h[idx] = new_val_minus
            r_minus = _compute_ratio_fast(h)
            
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio and new_val_minus >= 0:
                h[idx] = new_val_minus
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    i = np.arange(N)
    t = i / N
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform baseline
    candidates.append(np.ones(N))
    
    # 2. Disjoint pulses (every other)
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    candidates.append(h_alt)
    
    # 3. Disjoint pulses (every 3rd)
    h_alt3 = np.zeros(N)
    h_alt3[0::3] = 1.0
    candidates.append(h_alt3)
    
    # 4. Cosine bell
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 5. Half cosine (zero at edges)
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))
    
    # 6. Gaussian-like
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 7. Triangular
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 8. Power law shapes
    for p in [0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))
    
    # 9. Sinc-like
    candidates.append(np.sinc(2 * t - 1))
    
    # 10. Slight variations around uniform
    for amp in [0.01, 0.02, 0.05, 0.1, 0.15]:
        for freq in [1, 2, 3]:
            candidates.append(1.0 + amp * np.cos(2 * np.pi * freq * t))
    
    # 11. Bimodal / multi-peak shapes
    for n_peaks in [2, 3, 4]:
        centers = np.linspace(0.1, 0.9, n_peaks)
        for width in [0.08, 0.12, 0.18]:
            h = np.zeros(N)
            for c in centers:
                h += np.exp(-0.5 * ((t - c) / width) ** 2)
            candidates.append(h)
    
    # 12. Random smooth candidates
    np.random.seed(42)
    for _ in range(15):
        h_rand = np.random.rand(N)
        kernel = np.ones(7) / 7
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        h_smooth = np.clip(h_smooth, 0, None)
        if np.sum(h_smooth) > 1e-15:
            candidates.append(h_smooth)
    
    # 13. Random Fourier candidates
    np.random.seed(99)
    for _ in range(15):
        n_modes = np.random.randint(3, 10)
        coefs = np.zeros(2 * n_modes + 1)
        coefs[0] = np.random.uniform(0.3, 1.0)
        coefs[1:] = np.random.randn(2 * n_modes) * 0.2
        h_fourier = _fourier_to_heights(coefs, N, n_modes)
        if np.sum(h_fourier) > 1e-15:
            candidates.append(h_fourier)
    
    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = _compute_ratio_fast(best_h)
    
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = _compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Phase 1: Gradient Descent in Fourier space
    h_best, r_best = _fourier_gradient_descent(best_h, deadline, n_modes=20, max_iter=1000)
    if r_best < best_ratio:
        best_ratio = r_best
        best_h = h_best.copy()
        
    # Try a few other starts
    candidate_ratios = []
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = _compute_ratio_fast(h)
        candidate_ratios.append((r, h))
    candidate_ratios.sort(key=lambda x: x[0])
    
    for r, h in candidate_ratios[1:4]:
        if time.time() > deadline - 30.0:
            break
        h_f, r_f = _fourier_gradient_descent(h, deadline, n_modes=15, max_iter=500)
        if r_f < best_ratio:
            best_ratio = r_f
            best_h = h_f.copy()
            
    # Phase 2: Coordinate descent polish
    best_h, best_ratio = _coordinate_descent(best_h, deadline, max_iter=100)
    
    # Final safety clip
    best_h = np.clip(best_h, 0.0, None)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
