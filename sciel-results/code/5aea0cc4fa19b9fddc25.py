"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
W = 1.0 / (2.0 * N)  # interval width = 0.01

def _score(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = W
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    # Compute autoconvolution peak efficiently using FFT
    # Support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2]
    # Use fine grid for accurate peak detection
    n_grid = 2048
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    # Build f on grid
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(n) + 0.5) * w
    for i in range(n):
        lo = centers[i] - w/2
        hi = centers[i] + w/2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    # Convolve via FFT
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

def _score_fast(h):
    """Faster scoring for optimization - uses coarser grid."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = W
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    n_grid = 1024
    t = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    dt = t[1] - t[0]
    
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(n) + 0.5) * w
    for i in range(n):
        lo = centers[i] - w/2
        hi = centers[i] + w/2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h[i]
    
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score(best_h)
    
    # Generate diverse starting candidates
    candidates = []
    
    # Single spikes at various positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes close together
    for i in range(0, N-1, 3):
        for j in [i+1, i+2, i+3]:
            if j < N:
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = 1.0
                candidates.append(h)
    
    # Small triangular profiles
    for width in [2, 3, 4, 5]:
        half = width // 2
        for center in range(half, N - half):
            h = np.zeros(N)
            for i in range(N):
                dist = abs(i - center)
                if dist <= half:
                    h[i] = 1.0 - dist / (half + 0.5)
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Gaussian profiles
    for sigma in [1.0, 1.5, 2.0, 3.0]:
        for center in [N//4, N//2, 3*N//4]:
            i = np.arange(N)
            h = np.exp(-0.5 * ((i - center) / sigma)**2)
            candidates.append(h)
    
    # Evaluate candidates with fast scoring
    for h in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_fast(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best candidate using Nelder-Mead
    if time.time() < deadline - 25:
        def neg_score(h):
            h = np.abs(h)
            return -_score_fast(h)
        
        try:
            result = minimize(neg_score, best_h, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
        
        # Second round from a different starting point: single spike with small neighbors
        if time.time() < deadline - 15:
            h_start = np.zeros(N)
            h_start[N//2] = 1.0
            h_start[N//2 - 1] = 0.3
            h_start[N//2 + 1] = 0.3
            try:
                result = minimize(neg_score, h_start, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
                h_opt = np.abs(result.x)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
            except Exception:
                pass
        
        # Third round: try a wider Gaussian start
        if time.time() < deadline - 10:
            h_start = np.zeros(N)
            i = np.arange(N)
            h_start = np.exp(-0.5 * ((i - N//2) / 2.0)**2)
            try:
                result = minimize(neg_score, h_start, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
                h_opt = np.abs(result.x)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        def neg_score_precise(h):
            h = np.abs(h)
            return -_score(h)
        
        try:
            result = minimize(neg_score_precise, best_h, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Ensure nonnegativity and return
    best_h = np.abs(best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
