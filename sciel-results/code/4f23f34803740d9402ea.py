import numpy as np
import time
from scipy.optimize import minimize

N = 50
W = 1.0 / (2.0 * N)

def _find_sidon_set(n, target_size):
    """Find a Sidon set of size target_size in {0, ..., n-1}."""
    # Greedy construction
    sidon = []
    for i in range(n):
        can_add = True
        # Check if i can be added
        # A Sidon set requires all pairwise sums to be distinct.
        # If we add i, new sums are i + s for s in sidon.
        # These must not collide with existing sums or each other.
        
        # Efficient check:
        # Existing sums: S = {a+b | a,b in sidon, a<=b}
        # New sums: N = {i+s | s in sidon} U {2i}
        # We need N ∩ S = empty and N has no duplicates.
        
        # Since sidon is small, we can just generate sums.
        existing_sums = set()
        for a in range(len(sidon)):
            for b in range(a, len(sidon)):
                existing_sums.add(sidon[a] + sidon[b])
        
        new_sums = set()
        for s in sidon:
            new_sums.add(i + s)
        new_sums.add(2*i)
        
        # Check collisions
        if existing_sums & new_sums:
            can_add = False
        
        # Check duplicates in new_sums (only possible if i+s1 == i+s2 => s1==s2, so no duplicates in new_sums from distinct s)
        # Wait, if i+s1 == 2i => s1 == i. But i not in sidon yet.
        # So new_sums are distinct.
        
        if can_add:
            sidon.append(i)
            if len(sidon) >= target_size:
                break
                
    return sidon

def _score(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    w = W
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    # Compute autoconvolution peak
    # Use FFT for speed
    n_grid = 2048
    dt = 1.0 / n_grid
    
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

def _neg_ratio_smooth(x, temp=0.01):
    """Negative smooth ratio for minimization."""
    h = np.exp(x)
    # Normalize to L1=1 to remove scale invariance and focus on shape
    h_norm = h / np.sum(h)
    
    l2_sq = np.sum(h_norm**2) * W
    l1 = np.sum(h_norm) * W # This is W
    
    # Compute conv
    n_grid = 1024
    dt = 1.0 / n_grid
    f_grid = np.zeros(n_grid)
    centers = -0.25 + (np.arange(N) + 0.5) * W
    for i in range(N):
        lo = centers[i] - W/2
        hi = centers[i] + W/2
        idx_lo = int(round((lo + 0.5) / dt))
        idx_hi = int(round((hi + 0.5) / dt))
        idx_lo = max(0, min(idx_lo, n_grid))
        idx_hi = max(0, min(idx_hi, n_grid))
        if idx_hi > idx_lo:
            f_grid[idx_lo:idx_hi] += h_norm[i]
    
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    # Soft max
    shifted = conv - np.max(conv)
    max_conv_smooth = np.max(conv) + temp * np.log(np.sum(np.exp(shifted / temp)))
    
    if max_conv_smooth < 1e-15:
        return 0.0
    
    return -(l2_sq / (l1 * max_conv_smooth))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Find a good Sidon set
    # Try to find a set of size 8, 9, 10
    best_h = np.zeros(N)
    best_score = 0.0
    
    for k in [8, 9, 10, 11, 12]:
        sidon = _find_sidon_set(N, k)
        if len(sidon) < k:
            continue
        
        h0 = np.zeros(N)
        for idx in sidon:
            h0[idx] = 1.0
        
        # Normalize
        h0 = h0 / np.sum(h0)
        
        # Initial score
        s0 = _score(h0)
        if s0 > best_score:
            best_score = s0
            best_h = h0.copy()
        
        # Optimize
        if time.time() > deadline - 20:
            break
            
        x0 = np.log(np.maximum(h0, 1e-10))
        
        try:
            result = minimize(_neg_ratio_smooth, x0, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.sum(h_opt)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5:
        x_best = np.log(np.maximum(best_h, 1e-10))
        try:
            result = minimize(_neg_ratio_smooth, x_best, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.sum(h_opt)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return list(best_h)
