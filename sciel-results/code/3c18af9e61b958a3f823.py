"""
Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution.

Key insight: score = sum(h^2) / (delta * sum(h) * max(conv(h,h)))
With L1=1: score = sum(h^2) / (delta * max(conv))

For well-separated peaks, max(conv) ≈ max(h_i^2), so score ≈ (sum h_i^2) / (delta * max(h_i^2))
If k equal peaks: score = k/delta = k*N

Strategy: Use many well-separated peaks, then optimize heights via gradient-based methods.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _score_raw(h):
    """Raw score for internal comparison."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (DELTA * l1 * max_conv)

def _negative_score(h):
    """Negative score for minimization."""
    return -_score_raw(h)

def _score_normalized(h):
    """Score with L1 normalized to 1."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Try sparse configurations with many well-separated peaks
    # Use every other position for maximum separation
    for gap in [1, 2, 3]:
        for start in range(gap):
            if time.time() > deadline:
                return best_h if best_h is not None else np.ones(N)
            h = np.zeros(N)
            count = 0
            for i in range(start, N, gap):
                h[i] = 1.0
                count += 1
            if count > 0:
                update_best(h)
    
    # Phase 2: Random sparse configurations with varying densities
    np.random.seed(42)
    for trial in range(3000):
        if time.time() > deadline:
            break
        # Random number of peaks from 5 to 30
        num_peaks = np.random.randint(5, 31)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        # Random heights
        h[indices] = np.random.uniform(0.3, 2.0, size=num_peaks)
        update_best(h)
    
    # Phase 3: Gradient-based optimization from multiple starting points
    if time.time() < deadline - 30:
        # Start from best found so far
        if best_h is not None:
            h_start = best_h.copy()
            if np.sum(h_start) == 0:
                h_start = np.ones(N) / N
            
            # Optimize with SLSQP
            try:
                result = minimize(
                    _negative_score,
                    h_start,
                    method='SLSQP',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                )
                if result.success or result.fun < best_score:
                    h_opt = np.maximum(0, result.x)
                    update_best(h_opt)
            except Exception:
                pass
        
        # Try optimization from different random starts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random sparse start
            num_peaks = np.random.randint(5, 25)
            indices = np.random.choice(N, num_peaks, replace=False)
            h_start = np.zeros(N)
            h_start[indices] = np.random.uniform(0.5, 1.5, size=num_peaks)
            
            try:
                result = minimize(
                    _negative_score,
                    h_start,
                    method='SLSQP',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
                )
                if result.success or result.fun < best_score:
                    h_opt = np.maximum(0, result.x)
                    update_best(h_opt)
            except Exception:
                pass
    
    # Phase 4: Local refinement - try adjusting individual peaks
    if best_h is not None and time.time() < deadline - 5:
        h_curr = best_h.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            nonzero_idx = np.where(h_curr > 1e-8)[0]
            
            for idx in nonzero_idx:
                if time.time() > deadline:
                    break
                
                orig_val = h_curr[idx]
                # Try scaling
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
                    h_try = h_curr.copy()
                    h_try[idx] = orig_val * factor
                    s = _score_raw(h_try)
                    if s > best_score:
                        best_score = s
                        best_h = h_try.copy()
                        h_curr = h_try.copy()
                        improved = True
                        break
                
                # Try moving
                for offset in [-2, -1, 1, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-8:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        s = _score_raw(h_try)
                        if s > best_score:
                            best_score = s
                            best_h = h_try.copy()
                            h_curr = h_try.copy()
                            improved = True
                            break
            
            # Try adding a new peak
            if not improved:
                zero_idx = np.where(h_curr < 1e-8)[0]
                if len(zero_idx) > 0:
                    new_idx = np.random.choice(zero_idx)
                    h_try = h_curr.copy()
                    h_try[new_idx] = 0.5
                    s = _score_raw(h_try)
                    if s > best_score:
                        best_score = s
                        best_h = h_try.copy()
                        h_curr = h_try.copy()
            
            # Try removing a peak
            if not improved and len(nonzero_idx) > 1:
                remove_idx = np.random.choice(nonzero_idx)
                h_try = h_curr.copy()
                h_try[remove_idx] = 0.0
                s = _score_raw(h_try)
                if s > best_score:
                    best_score = s
                    best_h = h_try.copy()
                    h_curr = h_try.copy()
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
