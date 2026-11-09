import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    """Calculate the maximum overlap of h with shifted (1-h)."""
    max_overlap = 0.0
    for k in range(N):
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlap_k = np.sum(h_part * (1.0 - h_shifted))
        if overlap_k > max_overlap:
            max_overlap = overlap_k
    return max_overlap

def _objective(x):
    h = x
    sum_h = np.sum(h)
    target = N / 2.0
    sum_penalty = 1000.0 * abs(sum_h - target)
    
    below = np.sum(np.maximum(0, -h))
    above = np.sum(np.maximum(0, h - 1.0))
    bound_penalty = 1000.0 * (below + above)
    
    max_overlap = 0.0
    for k in range(N):
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlap_k = np.sum(h_part * (1.0 - h_shifted))
        if overlap_k > max_overlap:
            max_overlap = overlap_k
    
    return max_overlap + sum_penalty + bound_penalty

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Best known analytic solution: sinusoid
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    best_h = 0.5 + 0.5 * np.sin(t)
    best_score = _score(best_h)
    
    # Try other frequencies
    for freq in [1, 2, 3, 5, 10]:
        h_cand = 0.5 + 0.5 * np.sin(freq * t)
        s = _score(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
            
    # Refine with L-BFGS-B
    bounds = [(0.0, 1.0)] * N
    x0 = best_h.copy()
    
    def _obj_smooth(x):
        h = x
        sum_penalty = 500.0 * (np.sum(h) - N/2.0)**2
        below = np.sum(np.maximum(0, -h))
        above = np.sum(np.maximum(0, h - 1.0))
        bound_penalty = 500.0 * (below + above)**2
        
        overlaps = []
        for k in range(N):
            overlaps.append(np.sum(h[:N-k] * (1.0 - h[k:])))
        overlaps = np.array(overlaps)
        max_val = np.max(overlaps)
        soft_max = max_val + 0.05 * np.log(np.sum(np.exp((overlaps - max_val) / 0.05)))
        
        return soft_max + sum_penalty + bound_penalty

    if time.time() < deadline:
        try:
            result = minimize(_obj_smooth, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            h_new = np.clip(result.x, 0, 1)
            diff = N/2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new.copy()
        except Exception:
            pass

    return best_h
