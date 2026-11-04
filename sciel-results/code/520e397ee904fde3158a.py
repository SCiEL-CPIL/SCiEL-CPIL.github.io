"""Optimized program for the second autocorrelation inequality task."""
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

def _smooth_max(conv, temp=0.001):
    """Soft-max approximation of the maximum using log-sum-exp."""
    # Shift for numerical stability
    shifted = conv - np.max(conv)
    log_sum_exp = np.max(shifted) + np.log(np.sum(np.exp(shifted / temp))) * temp
    return log_sum_exp

def _ratio_smooth(h, temp=0.001):
    """Smooth version of the ratio using soft-max for the autoconvolution peak."""
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
    
    # Use soft-max for smoothness
    max_conv_smooth = _smooth_max(conv, temp)
    
    if max_conv_smooth < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv_smooth)

def _neg_ratio_smooth(x, temp=0.001):
    """Negative smooth ratio for minimization with log-parameterization."""
    h = np.exp(x)  # h_i = exp(x_i) ensures non-negativity
    return -_ratio_smooth(h, temp)

def _neg_ratio_log(x):
    """Negative ratio for minimization with log-parameterization."""
    h = np.exp(x)
    return -_score_fast(h)

def _get_sidon_set(n, k):
    """Generate a Sidon set of size k from {0, ..., n-1}."""
    # Simple greedy Sidon set construction
    sidon = []
    for i in range(n):
        can_add = True
        for j in range(len(sidon)):
            for m in range(len(sidon)):
                if j == m:
                    continue
                # Check if i - sidon[j] == sidon[m] - sidon[j] or similar
                if i - sidon[j] == sidon[m] - sidon[j]:
                    can_add = False
                    break
                if i - sidon[m] == sidon[j] - sidon[j]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            sidon.append(i)
        if len(sidon) >= k:
            break
    return sidon

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = _score(best_h)
    
    # Strategy 1: Log-parameterized optimization with smooth surrogate
    # Use soft-max to smooth the non-differentiable max operation
    
    # Generate diverse starting points in log-space
    log_starts = []
    
    # Single spike
    x0 = np.full(N, -10.0)
    x0[N // 2] = 0.0
    log_starts.append(x0)
    
    # Sidon set (low autocorrelation peaks)
    sidon = _get_sidon_set(N, 8)
    if len(sidon) >= 4:
        x_sidon = np.full(N, -10.0)
        for idx in sidon[:8]:
            x_sidon[idx] = 0.0
        log_starts.append(x_sidon)
    
    # Symmetric patterns
    for width in [3, 5, 7]:
        half = width // 2
        for center in [N//4, N//2, 3*N//4]:
            x_sym = np.full(N, -10.0)
            for i in range(N):
                dist = abs(i - center)
                if dist <= half:
                    x_sym[i] = -dist * 0.5
            log_starts.append(x_sym)
    
    # Gaussian in log-space
    for sigma in [1.5, 2.5, 4.0]:
        i = np.arange(N)
        x_gauss = -0.5 * ((i - N//2) / sigma)**2 - 5.0
        log_starts.append(x_gauss)
    
    # Evaluate and optimize from each starting point
    for x0 in log_starts:
        if time.time() > deadline - 30:
            break
        
        # First: quick evaluation
        h_eval = np.exp(x0)
        s_eval = _score_fast(h_eval)
        if s_eval > best_score:
            best_score = s_eval
            best_h = h_eval.copy()
        
        # Optimize using smooth surrogate
        if time.time() < deadline - 20:
            try:
                result = minimize(_neg_ratio_smooth, x0, method='L-BFGS-B',
                               options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8,
                                        'maxls': 50})
                h_opt = np.exp(result.x)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Second: refine with Nelder-Mead in original space
        if time.time() < deadline - 15:
            h_current = np.exp(x0)
            if s_eval > best_score * 0.9:  # Only if promising
                try:
                    def neg_score_nm(h):
                        h = np.abs(h)
                        return -_score_fast(h)
                    
                    result = minimize(neg_score_nm, h_current, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                            'adaptive': True})
                    h_opt = np.abs(result.x)
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # Strategy 2: Direct optimization from best found so far
    if time.time() < deadline - 20:
        x_best = np.log(np.maximum(best_h, 1e-10))
        
        # Smooth optimization with tighter tolerance
        try:
            result = minimize(_neg_ratio_smooth, x_best, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxls': 50})
            h_opt = np.exp(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish with exact scoring
    if time.time() < deadline - 10:
        try:
            def neg_score_precise(h):
                h = np.abs(h)
                return -_score(h)
            
            result = minimize(neg_score_precise, best_h, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 4: Try a few more random restarts if time permits
    if time.time() < deadline - 15:
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random log-normal distribution
            x_rand = np.random.randn(N) * 1.5 - 5.0
            h_rand = np.exp(x_rand)
            h_rand = h_rand / np.sum(h_rand) * 5.0  # Normalize
            
            s_rand = _score_fast(h_rand)
            if s_rand > best_score:
                best_score = s_rand
                best_h = h_rand.copy()
            
            # Quick optimization from random start
            if time.time() < deadline - 8:
                try:
                    result = minimize(_neg_ratio_smooth, x_rand, method='L-BFGS-B',
                                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8,
                                            'maxls': 30})
                    h_opt = np.exp(result.x)
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # Final polish
    if time.time() < deadline - 3:
        try:
            def neg_score_final(h):
                h = np.abs(h)
                return -_score(h)
            
            result = minimize(neg_score_final, best_h, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and return
    best_h = np.abs(best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
