# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten for sorting
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort sums and get corresponding probabilities
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Precompute thresholds: thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find the number of pair_sums strictly less than it
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Create safe index array for cum_probs lookup
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    
    # P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def _hill_climb_weights(s, w0, max_steps=5000, step_size=0.1, deadline=None):
    """
    Simple coordinate descent / hill climbing on the simplex for weights.
    Much faster per step than Nelder-Mead for weight-only optimization.
    """
    n = len(s)
    w = w0.copy()
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    
    best_P = _compute_P_fast(s, w)
    best_w = w.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try moving probability mass from i to j
            for j in range(n):
                if i == j:
                    continue
                if w[i] <= 1e-10:
                    continue
                
                # Try several step sizes
                for frac in [0.05, 0.1, 0.2, 0.5]:
                    delta = w[i] * frac
                    if delta > w[i]:
                        continue
                    
                    w_new = w.copy()
                    w_new[i] -= delta
                    w_new[j] += delta
                    w_new = np.maximum(w_new, 0)
                    w_sum = w_new.sum()
                    if w_sum <= 1e-12:
                        continue
                    w_new = w_new / w_sum
                    
                    P_new = _compute_P_fast(s, w_new)
                    if P_new > best_P + 1e-12:
                        best_P = P_new
                        best_w = w_new.copy()
                        w = w_new.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    # Generate diverse support structures
    supports_to_try = []
    
    # 1. Geometric progressions with various bases and lengths
    for n in [3, 4, 5, 6, 7, 8, 10, 12]:
        for base in [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            s = np.array([base**i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
    
    # 2. Arithmetic progressions
    for n in [3, 4, 5, 6, 8, 10, 12, 15]:
        for step in [0.1, 0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
    
    # 3. Two-point measures: {0, a} for various a
    for a in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        supports_to_try.append(np.array([0.0, a], dtype=np.float64))
    
    # 4. Three-point: {0, a, b}
    for a in [1.0, 2.0, 5.0]:
        for b in [3.0, 5.0, 10.0, 20.0, 50.0]:
            if b > a:
                supports_to_try.append(np.array([0.0, a, b], dtype=np.float64))
    
    # 5. Mixed: small cluster + large jumps
    for n in [4, 5, 6, 8]:
        for k in range(2, n):
            for scale in [5.0, 10.0, 20.0]:
                s = np.zeros(n, dtype=np.float64)
                s[:k] = np.arange(k, dtype=np.float64)
                for j in range(k, n):
                    s[j] = scale * (j - k + 1)
                supports_to_try.append(s)
    
    # 6. Power-law: {i^alpha}
    for n in [4, 5, 6, 8, 10]:
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            s = np.array([i**alpha for i in range(1, n+1)], dtype=np.float64)
            supports_to_try.append(s)
    
    # 7. Log-spaced
    for n in [4, 5, 6, 8, 10, 12]:
        for lo, hi in [(0, 10), (0, 50), (0, 100), (1, 1000)]:
            s = np.logspace(lo, hi, n)
            supports_to_try.append(s)
    
    # 8. Random sorted supports
    np.random.seed(12345)
    for trial in range(50):
        n = np.random.randint(3, 11)
        s_raw = np.sort(np.random.uniform(0, 100, n))
        # Ensure minimum spacing
        s = s_raw.copy()
        for i in range(1, n):
            if s[i] - s[i-1] < 0.01:
                s[i] = s[i-1] + 0.01
        supports_to_try.append(s)
    
    # Optimize weights for each support using hill climbing
    for idx, s in enumerate(supports_to_try):
        if time.time() > deadline - 30.0:
            break
        
        n = len(s)
        
        # Multiple initial weight configurations
        initial_weights = [
            np.ones(n) / n,  # Uniform
            np.array([0.5] + [0.5/(n-1)]*(n-1)),  # Heavy on first
            np.array([0.5/(n-1)]*(n-1) + [0.5]),  # Heavy on last
        ]
        
        if n >= 3:
            # Geometric weights
            r = 0.7
            w_geom = np.array([r**i for i in range(n)])
            w_geom = w_geom / w_geom.sum()
            initial_weights.append(w_geom)
            
            # Inverse geometric
            r = 0.7
            w_inv = np.array([r**(n-1-i) for i in range(n)])
            w_inv = w_inv / w_inv.sum()
            initial_weights.append(w_inv)
        
        # Try each initial configuration
        for w0 in initial_weights:
            if time.time() > deadline - 30.0:
                break
            
            # Hill climbing
            w_opt, P_opt = _hill_climb_weights(s, w0, max_steps=2000, deadline=deadline - 30.0)
            
            if P_opt > best_P:
                best_P = P_opt
                best_support = s.copy()
                best_weights = w_opt.copy()
        
        # Also try Nelder-Mead for small n (n <= 6) as it can escape local optima
        if n <= 6 and time.time() < deadline - 60.0:
            def objective_softmax(z):
                exp_z = np.exp(z - np.max(z))
                w = exp_z / exp_z.sum()
                P = _compute_P_fast(s, w)
                return -P
            
            z0 = np.zeros(n)
            try:
                res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < -best_P:
                    z_opt = res.x
                    exp_z = np.exp(z_opt - np.max(z_opt))
                    w_opt = exp_z / exp_z.sum()
                    P_opt = -res.fun
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = s.copy()
                        best_weights = w_opt.copy()
            except Exception:
                pass
    
    # Final refinement: if we have time, do more hill-climbing steps on the best solution
    if time.time() < deadline - 10.0:
        remaining_steps = 5000
        best_weights, best_P = _hill_climb_weights(best_support, best_weights, 
                                                    max_steps=remaining_steps, 
                                                    deadline=deadline - 10.0)
    
    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    best_support = np.abs(best_support)
    best_weights = np.abs(best_weights)
    
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
