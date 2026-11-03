"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

The autoconvolution of the step function f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t) with delta = 1/50:
- ||f||_1 = delta * sum(h_i)
- ||f||_2^2 = delta * sum(h_i^2)
- ||f*f||_inf = delta * max(conv(h,h))

Ratio = (delta * sum h^2) / (delta * sum h * delta * max conv) = sum h^2 / (delta * sum h * max conv)

The evaluator normalizes (likely L1=1), so we maximize: sum(h^2) / (delta * max(conv(h,h)))
subject to sum(h)=1, h>=0.

Strategy: Use sparse, well-separated peaks to minimize autoconvolution overlap.
"""
import numpy as np
import time
from itertools import combinations

N = 50
DELTA = 1.0 / N  # = 0.02

def _compute_score(h):
    """Compute the score assuming L1 normalization."""
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

def _score_raw(h):
    """Raw score without assuming normalization - used for internal comparison."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    # The actual ratio: sum h^2 / (delta * sum h * max conv)
    return l2_sq / (DELTA * l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Try sparse configurations with few peaks
    # The key insight: fewer peaks with larger values -> higher sum(h^2)/sum(h) ratio
    # But we need to minimize max autoconvolution
    
    # Single peak (baseline)
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        update_best(h)
    
    # Two peaks
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline:
                return best_h if best_h is not None else np.ones(N)
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            update_best(h)
    
    # Three peaks - try well-separated positions
    for i in range(0, N, 3):
        for j in range(i+5, N, 3):
            for k in range(j+5, N, 3):
                if time.time() > deadline:
                    return best_h if best_h is not None else np.ones(N)
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = 1.0
                h[k] = 1.0
                update_best(h)
    
    # Phase 2: Random sparse configurations
    np.random.seed(42)
    for _ in range(5000):
        if time.time() > deadline:
            break
        num_peaks = np.random.randint(2, 8)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        h[indices] = np.random.uniform(0.5, 2.0, size=num_peaks)
        update_best(h)
    
    # Phase 3: Local refinement of best configuration
    if best_h is not None:
        h_curr = best_h.copy()
        for _ in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            # Try perturbing each nonzero entry
            nonzero_idx = np.where(h_curr > 1e-6)[0]
            for idx in nonzero_idx:
                if time.time() > deadline:
                    break
                
                orig_val = h_curr[idx]
                # Try scaling up and down
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
                
                # Try moving the peak
                for offset in [-2, -1, 1, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
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
                zero_idx = np.where(h_curr < 1e-6)[0]
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
    
    # Phase 4: Try structured patterns
    # Alternating peaks
    for start in range(N):
        for gap in [2, 3, 4, 5, 7, 10]:
            if time.time() > deadline:
                break
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                update_best(h)
    
    # Gaussian-like with varying widths
    for center in range(5, N-5):
        for width in [2, 3, 4, 5, 7, 10, 15]:
            if time.time() > deadline:
                break
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width)**2)
            if np.sum(h) > 0:
                update_best(h)
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
