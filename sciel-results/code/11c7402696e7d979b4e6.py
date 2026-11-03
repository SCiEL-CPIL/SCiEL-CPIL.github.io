import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/1/4]

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    where AutoConv is the continuous autoconvolution of the step function.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _coordinate_descent(h, max_iter=200, tol=1e-10):
    """
    Coordinate descent to maximize L2^2 / (L1 * AutoConvMax).
    We fix L1=1 by normalizing, so we maximize L2^2 / AutoConvMax.
    """
    h = h.copy()
    h = np.maximum(0, h)
    l1 = np.sum(h)
    if l1 == 0:
        return h
    
    # Normalize to L1=1
    h = h / l1
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try setting h[i] to various values
            best_val = h[i]
            best_score = _score_raw(h)
            
            # Try a few candidate values for h[i]
            candidates = [0.0, h[i] * 0.5, h[i] * 0.75, h[i] * 1.25, h[i] * 1.5, 0.01, 0.05, 0.1]
            
            for val in candidates:
                h[i] = val
                h = np.maximum(0, h)
                l1_new = np.sum(h)
                if l1_new == 0:
                    h[i] = best_val
                    continue
                h_norm = h / l1_new
                s = _score_raw(h_norm)
                if s > best_score + tol:
                    best_score = s
                    best_val = val
            
            h[i] = best_val
            h = np.maximum(0, h)
            l1_new = np.sum(h)
            if l1_new > 0:
                h = h / l1_new
            if abs(best_val - h[i]) > tol:
                improved = True
        
        if not improved:
            break
    
    return h

def _greedy_support_search(h_init, max_supports=50):
    """
    Greedy search over support sets. For each support size k,
    try to find a good subset of indices to place mass on.
    """
    best_h = h_init.copy()
    best_score = _score_raw(h_init)
    
    x_idx = np.arange(N)
    
    # Try different support sizes
    for k in range(3, min(N, 20)):
        if time.time() > 140:
            break
        
        # Start with a random subset
        best_subset_score = 0
        best_subset = None
        
        for trial in range(20):
            if time.time() > 140:
                break
            
            # Random subset
            subset = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[subset] = 1.0
            h = _coordinate_descent(h, max_iter=50)
            s = _score_raw(h)
            
            if s > best_subset_score:
                best_subset_score = s
                best_subset = subset.copy()
                best_h = h.copy()
        
        if best_subset is not None and best_subset_score > best_score:
            best_score = best_subset_score
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Parametric initializations ---
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline - 20: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # --- Phase 2: Coordinate Descent from best initialization ---
    
    if time.time() < deadline - 30:
        h_cd = _coordinate_descent(best_h, max_iter=300)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd.copy()
    
    # --- Phase 3: Greedy support search ---
    
    if time.time() < deadline - 20:
        h_greedy, s_greedy = _greedy_support_search(best_h, max_supports=30)
        if s_greedy > best_score:
            best_score = s_greedy
            best_h = h_greedy.copy()
    
    # --- Phase 4: Final polish with coordinate descent ---
    
    if time.time() < deadline - 5:
        h_final = _coordinate_descent(best_h, max_iter=500)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    return list(best_h)
