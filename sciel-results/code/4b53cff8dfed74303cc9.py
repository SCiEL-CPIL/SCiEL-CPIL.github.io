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
    
    # Find indices of first element >= threshold (i.e., count of elements < threshold)
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Calculate sum of probabilities for elements < threshold
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    
    # P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def _objective_joint(params, n):
    """
    Objective function for joint optimization of support and weights.
    params: [log_s_1, ..., log_s_{n-1}, log_w_0, ..., log_w_{n-1}]
    Support s_0 is fixed at 0.
    Weights are normalized via softmax.
    """
    # Unpack
    log_s = params[:n-1]
    log_w = params[n-1:]
    
    # Support: s_0 = 0, s_i = exp(log_s_{i-1})
    # Ensure strict increasing order: s_0 < s_1 < ... < s_{n-1}
    # We enforce this by construction: s_i = exp(cumsum(log_s) + offset) ? 
    # Simpler: just take exp and sort? No, sorting breaks the mapping to weights.
    # Better: enforce order in the parameter space.
    # Let's use: s_1 = exp(u_1), s_2 = s_1 + exp(u_2), ...
    # This ensures s_0=0 < s_1 < s_2 < ...
    
    s = np.zeros(n)
    if n > 1:
        # s[1] = exp(log_s[0])
        # s[i] = s[i-1] + exp(log_s[i-1]) for i > 1
        # This guarantees positive steps.
        for i in range(1, n):
            s[i] = s[i-1] + np.exp(log_s[i-1])
            
    # Weights: softmax
    max_log_w = np.max(log_w)
    exp_w = np.exp(log_w - max_log_w)
    w = exp_w / np.sum(exp_w)
    
    # Compute P
    P = _compute_P_fast(s, w)
    
    return -P

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
    
    # 1. Analytical 2-point solution as a strong baseline
    # Optimal q for {0, a} is root of 8q^3 - 15q^2 + 6q = 0 -> q ~ 0.579
    q_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([q_opt, 1.0 - q_opt])
    P_2pt = _compute_P_fast(s_2pt, w_2pt)
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = s_2pt.copy()
        best_weights = w_2pt.copy()

    # 2. Joint Optimization for n=3 and n=4
    # We optimize the "step sizes" in log-space and the weights in log-space.
    # This allows the support points to move freely while maintaining order.
    
    for n in [3, 4]:
        if time.time() > deadline - 120.0:
            break
            
        # Number of parameters: (n-1) for steps, n for weights
        n_params = (n - 1) + n
        
        # Initial guesses
        # Guess 1: Based on 2-point, add a small extra point
        # Guess 2: Geometric progression
        # Guess 3: Random
        
        initial_guesses = []
        
        # Guess 1: Uniform weights, geometric support
        # s = [0, 1, 2, 4] -> steps [1, 1, 2] -> logs [0, 0, 0.69]
        if n == 3:
            steps = np.array([1.0, 2.0])
        elif n == 4:
            steps = np.array([1.0, 1.0, 2.0])
        else:
            steps = np.ones(n-1)
        
        log_steps = np.log(steps)
        log_weights = np.zeros(n)
        initial_guesses.append(np.concatenate([log_steps, log_weights]))
        
        # Guess 2: Geometric base 1.5
        if n == 3:
            steps = np.array([1.0, 0.5]) # s=[0,1,1.5]
        elif n == 4:
            steps = np.array([1.0, 0.5, 0.75]) # s=[0,1,1.5,2.25]
        log_steps2 = np.log(steps)
        log_weights2 = np.zeros(n)
        initial_guesses.append(np.concatenate([log_steps2, log_weights2]))
        
        # Guess 3: Based on 2-point optimum, split the mass
        # If we had {0, 1}, maybe try {0, 0.5, 1.5}
        if n == 3:
            steps = np.array([0.5, 1.0])
        elif n == 4:
            steps = np.array([0.25, 0.5, 1.0])
        log_steps3 = np.log(steps)
        # Weights biased towards 0
        w_init = np.zeros(n)
        w_init[0] = 1.5
        w_init[1:] = 1.0
        log_weights3 = np.log(w_init)
        initial_guesses.append(np.concatenate([log_steps3, log_weights3]))
        
        for z0 in initial_guesses:
            if time.time() > deadline - 60.0:
                break
                
            try:
                res = minimize(_objective_joint, z0, args=(n,), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                if res.success:
                    P_opt = -res.fun
                    if P_opt > best_P:
                        # Reconstruct support and weights
                        log_s = res.x[:n-1]
                        log_w = res.x[n-1:]
                        
                        s = np.zeros(n)
                        for i in range(1, n):
                            s[i] = s[i-1] + np.exp(log_s[i-1])
                            
                        max_log_w = np.max(log_w)
                        exp_w = np.exp(log_w - max_log_w)
                        w = exp_w / np.sum(exp_w)
                        
                        best_P = P_opt
                        best_support = s.copy()
                        best_weights = w.copy()
                        
            except Exception as e:
                pass

    # 3. Fallback: Grid search / Optimization on fixed supports (from current program)
    # This ensures we don't lose the gains from the previous version if the joint opt fails.
    if time.time() < deadline - 30.0:
        supports_to_try = []
        # Geometric
        for n in [3, 4, 5]:
            for base in [1.5, 2.0, 3.0]:
                s = np.array([base**i for i in range(n)])
                supports_to_try.append(s)
        
        for s in supports_to_try:
            if time.time() > deadline - 10.0:
                break
            n = len(s)
            # Optimize weights only
            def obj_w(z):
                exp_z = np.exp(z - np.max(z))
                w = exp_z / exp_z.sum()
                return -_compute_P_fast(s, w)
            
            z0 = np.zeros(n)
            try:
                res = minimize(obj_w, z0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < -best_P:
                    exp_z = np.exp(res.x - np.max(res.x))
                    w_opt = exp_z / exp_z.sum()
                    P_opt = -res.fun
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = s.copy()
                        best_weights = w_opt.copy()
            except Exception:
                pass

    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
        
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
        
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
