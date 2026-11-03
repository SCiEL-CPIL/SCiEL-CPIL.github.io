"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
W = 1.0 / 1200.0

def compute_ratio(h, n_grid=4000):
    """Compute max(autoconv) / l2_sq for step function h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 1 << (2 * n_grid - 1).bit_length()
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
    deadline = time.time() + 130.0
    
    # Try several analytical shapes
    i = np.arange(N)
    t = i / N  # 0 to ~1
    
    candidates = []
    
    # 1. Uniform (baseline)
    candidates.append(np.ones(N))
    
    # 2. Cosine bell: 0.5*(1+cos(2*pi*t)) centered
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 3. Half cosine bell (zero at edges)
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))
    
    # 4. Gaussian-like
    sigma = 0.25
    candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 5. Raised cosine (Hann window)
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 6. Triangular peak
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 7. Power law: (1 - |2t-1|^p)
    for p in [0.5, 2.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))
    
    # 8. Sinc-like
    candidates.append(np.sinc(2 * t - 1))
    
    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = float('inf')
    
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r = compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local refinement: gradient descent on the best candidate
    h = best_h.copy()
    lr = 0.01
    for step in range(500):
        if time.time() > deadline:
            break
        r = compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
        
        # Numerical gradient (finite differences on a few random coordinates)
        n_coords = 50
        idxs = np.random.choice(N, size=n_coords, replace=False)
        eps = 0.01 * max(1.0, np.max(h))
        grad = np.zeros(N)
        for idx in idxs:
            h_old = h[idx]
            h[idx] = h_old + eps
            r_plus = compute_ratio(h)
            h[idx] = h_old - eps
            r_minus = compute_ratio(h)
            h[idx] = h_old
            grad[idx] = (r_plus - r_minus) / (2 * eps)
        
        h -= lr * grad
        h = np.clip(h, 0.0, None)
    
    # Final evaluation
    final_r = compute_ratio(best_h)
    if final_r < best_ratio:
        best_ratio = final_r
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
