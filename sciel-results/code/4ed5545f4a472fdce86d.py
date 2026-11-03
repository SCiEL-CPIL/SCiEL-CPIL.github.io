import numpy as np
import time

N = 600
W = 1.0 / 1200.0

# EVOLVE-BLOCK-START
def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT with grid resolution matched to step function for accuracy.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    # Use 4x oversampling: 2400 points for 600 intervals
    n_grid = 2400
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    # Next power of 2 >= 2*n_grid - 1
    n_fft = 1
    while n_fft < 2 * n_grid - 1:
        n_fft *= 2
    
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    i = np.arange(N)
    t = i / N  # t in [0, 1), maps to [-0.25, 0.25)
    
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
    
    # 6. Gaussian-like (various widths)
    for sigma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 7. Triangular
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 8. Power law shapes (sharp peaks)
    for p in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))
    
    # 9. Sinc-like
    candidates.append(np.sinc(2 * t - 1))
    
    # 10. Sharp peak family: concentrate mass near center
    # Parametric: h_i = exp(-alpha * |i - N/2|^beta)
    center = N // 2
    dist = np.abs(i - center) / (N / 2.0)  # normalized distance from center
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for beta in [1.0, 2.0, 4.0]:
            h_peak = np.exp(-alpha * dist ** beta)
            candidates.append(h_peak)
    
    # 11. Asymmetric shapes
    for asym in [0.3, 0.5, 0.7]:
        h_asym = np.exp(-0.5 * ((t - (0.5 - asym * 0.2)) / 0.15) ** 2)
        candidates.append(h_asym)
    
    # 12. Random smooth candidates
    np.random.seed(42)
    for _ in range(15):
        h_rand = np.random.rand(N)
        # Smooth with larger kernel
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        h_smooth = np.clip(h_smooth, 0, None)
        if np.sum(h_smooth) > 1e-15:
            candidates.append(h_smooth)
    
    # 13. More structured random: random Fourier modes
    np.random.seed(123)
    for _ in range(10):
        coeffs = np.random.randn(20) * 0.1
        h_fourier = 1.0 + sum(c * np.cos(2 * np.pi * k * t) for k, c in enumerate(coeffs, 1))
        h_fourier = np.clip(h_fourier, 0, None)
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
    
    # Local refinement using coordinate descent
    h = best_h.copy()
    max_h = np.max(h)
    step_size = max(0.005, max_h * 0.005)
    
    np.random.seed(7)
    for iteration in range(200):
        if time.time() > deadline - 15.0:
            break
        
        improved = False
        # Try perturbing a subset of coordinates
        n_trials = 40
        idxs = np.random.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            # Try increasing
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            
            # Try decreasing
            if h_old > step_size:
                h[idx] = h_old - step_size
                r_minus = _compute_ratio_fast(h)
            else:
                r_minus = float('inf')
            
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    # Final evaluation
    final_r = _compute_ratio_fast(h)
    if final_r < best_ratio:
        best_ratio = final_r
        best_h = h.copy()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
