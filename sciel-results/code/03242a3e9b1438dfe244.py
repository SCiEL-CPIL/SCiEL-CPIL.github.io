import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(p):
    p = p[p > 1e-12]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def _compute_score(P):
    # P is 10x10
    n = 10
    # Flatten indices
    ys, xs = np.indices((n, n))
    
    # D = H(x - y)
    diffs = xs - ys
    # Aggregate probabilities for each diff
    # diffs range from -9 to 9
    diff_counts = np.zeros(19)
    for d in range(-9, 10):
        mask = (diffs == d)
        diff_counts[d + 9] = np.sum(P[mask])
    D = _shannon_entropy(diff_counts)
    
    # M = max(H(x), H(x+y), H(x+2y))
    # H(x)
    x_counts = np.zeros(n)
    for x in range(n):
        x_counts[x] = np.sum(P[:, x])
    Hx = _shannon_entropy(x_counts)
    
    # H(x+y)
    # x+y range 0..18
    sum_counts = np.zeros(19)
    for s in range(19):
        mask = (xs + ys == s)
        sum_counts[s] = np.sum(P[mask])
    Hs = _shannon_entropy(sum_counts)
    
    # H(x+2y)
    # x+2y range 0..27
    sum2_counts = np.zeros(28)
    for v in range(28):
        mask = (xs + 2 * ys == v)
        sum2_counts[v] = np.sum(P[mask])
    Hv = _shannon_entropy(sum2_counts)
    
    M = max(Hx, Hs, Hv)
    
    if M <= 1e-9:
        return -np.inf
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    n = 10
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Helper to convert weights on support to P
    def support_to_P(support, weights):
        P = np.zeros((n, n))
        for i, (x, y) in enumerate(support):
            P[y, x] = weights[i]
        return P
    
    def objective(weights):
        P = support_to_P(support, weights)
        return -_compute_score(P)
    
    # 1. Generate candidate supports
    # We want 19 points with distinct differences.
    # We will try several random configurations.
    
    num_candidates = 50
    candidates = []
    
    for _ in range(num_candidates):
        if time.time() > deadline:
            break
        support = []
        # Shuffle diffs to ensure we pick one point per diff
        diffs = list(range(-9, 10))
        np.random.shuffle(diffs)
        for k in diffs:
            # valid x: max(0, k) <= x <= min(9, 9+k)
            x_min = max(0, k)
            x_max = min(9, 9 + k)
            if x_min > x_max:
                continue # Should not happen
            x = np.random.randint(x_min, x_max + 1)
            y = x - k
            support.append((x, y))
        
        # Check if we have 19 unique points
        if len(set(support)) != 19:
            continue
            
        # Initial score with uniform weights
        w = np.ones(19) / 19
        P = support_to_P(support, w)
        score = _compute_score(P)
        candidates.append((score, support))
        
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 5
    top_candidates = candidates[:5]
    
    # 2. Optimize weights for top candidates
    for score_init, support in top_candidates:
        if time.time() > deadline:
            break
        
        w0 = np.ones(19) / 19
        # Use Nelder-Mead
        try:
            res = minimize(objective, w0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'maxfev': 5000})
            if res.success or res.fun < -score_init:
                w_opt = res.x
                # Ensure non-negative and normalized
                w_opt = np.maximum(w_opt, 0)
                w_opt = w_opt / np.sum(w_opt)
                
                P_opt = support_to_P(support, w_opt)
                score_opt = _compute_score(P_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt
        except Exception:
            pass
            
    # 3. Also try a global random search on the full 100x100 grid?
    # Might be too slow. But we can try a few random sparse distributions.
    
    # If we haven't found anything better than uniform, return uniform.
    # But best_score starts at 0. Uniform score is ~0.87.
    # We should initialize best_P and best_score with uniform.
    
    P_uniform = np.ones((n, n)) / (n * n)
    score_uniform = _compute_score(P_uniform)
    
    if best_score < score_uniform:
        best_P = P_uniform
        best_score = score_uniform
        
    return best_P
