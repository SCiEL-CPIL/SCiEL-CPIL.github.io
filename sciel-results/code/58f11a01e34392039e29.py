# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import brentq

def _get_pooling_matrices():
    """
    Precompute sparse-like (but stored as dense for simplicity/small N) 
    pooling matrices for slopes -1, 0, 1, 2.
    M_s[k, i] = 1 if cell i maps to bin k for slope s.
    """
    n = 10
    y_idx, x_idx = np.indices((n, n))
    flat_y = y_idx.flatten()
    flat_x = x_idx.flatten()
    
    # Slope -1: x - y. Range -9..9. Offset by 9 -> 0..18.
    vals_m1 = (flat_x - flat_y + 9)
    bins_m1 = 19
    M_m1 = np.zeros((bins_m1, 100))
    M_m1[vals_m1, np.arange(100)] = 1.0
    
    # Slope 0: x. Range 0..9.
    vals_0 = flat_x
    bins_0 = 10
    M_0 = np.zeros((bins_0, 100))
    M_0[vals_0, np.arange(100)] = 1.0
    
    # Slope 1: x + y. Range 0..18.
    vals_1 = (flat_x + flat_y)
    bins_1 = 19
    M_1 = np.zeros((bins_1, 100))
    M_1[vals_1, np.arange(100)] = 1.0
    
    # Slope 2: x + 2y. Range 0..27.
    vals_2 = (flat_x + 2 * flat_y)
    bins_2 = 28
    M_2 = np.zeros((bins_2, 100))
    M_2[vals_2, np.arange(100)] = 1.0
    
    return M_m1, M_0, M_1, M_2

def _compute_entropies(weights, M_m1, M_0, M_1, M_2):
    """
    Compute entropies H_{-1}, H_0, H_1, H_2 given weights on active cells.
    weights: array of shape (k,)
    """
    # Aggregate weights into bins
    # Note: weights are already normalized to sum to 1 for the whole grid? 
    # In this strategy, we fix the support and solve for weights summing to 1.
    
    # H_{-1} (Numerator D)
    p_m1 = M_m1 @ weights
    p_m1 = p_m1[p_m1 > 0]
    H_m1 = -np.sum(p_m1 * np.log2(p_m1)) if len(p_m1) > 0 else 0.0
    
    # H_0, H_1, H_2 (Denominator M)
    p_0 = M_0 @ weights
    p_0 = p_0[p_0 > 0]
    H_0 = -np.sum(p_0 * np.log2(p_0)) if len(p_0) > 0 else 0.0
    
    p_1 = M_1 @ weights
    p_1 = p_1[p_1 > 0]
    H_1 = -np.sum(p_1 * np.log2(p_1)) if len(p_1) > 0 else 0.0
    
    p_2 = M_2 @ weights
    p_2 = p_2[p_2 > 0]
    H_2 = -np.sum(p_2 * np.log2(p_2)) if len(p_2) > 0 else 0.0
    
    M = max(H_0, H_1, H_2)
    
    return H_m1, M

def _solve_weights_for_support(active_indices, M_m1, M_0, M_1, M_2):
    """
    Given a set of active cell indices, find weights w_i > 0 summing to 1
    that maximize H_{-1} / max(H_0, H_1, H_2).
    
    We exploit the structure: 
    Let the active cells be S.
    For a fixed support S, the entropies are functions of the relative weights.
    Specifically, H(S) is scale invariant if we consider the distribution induced by P on the bins.
    Wait, H is defined on the pooled distribution.
    If we scale all weights by C, the pooled probabilities scale by C, but they must sum to 1.
    So if w sums to 1, p_bins = M @ w sums to 1.
    
    The objective R(w) = H_m1(w) / max(H_0(w), H_1(w), H_2(w)).
    
    Is this concave? H is concave. Ratio of concave is not necessarily concave.
    However, for small supports, we can use a 1D parameterization if the structure allows,
    or use a local optimizer on the simplex.
    
    Given the small k (support size), we can use scipy.optimize.minimize with method='SLSQP'
    or 'Nelder-Mead' on the (k-1) dimensional simplex.
    Since k is small (e.g., <= 20), this is fast.
    
    To speed up, we can also try a heuristic: 
    Often, uniform weight on the support is a good local max.
    Then refine.
    """
    k = len(active_indices)
    if k == 0:
        return None, 0.0
    if k == 1:
        # Single cell: H=0 for all. Ratio 0/0 -> invalid or 0.
        # Actually if one cell, H_m1=0, M=0. 
        # The task says M <= 1e-9 is invalid.
        # So single cell is bad.
        return None, 0.0
        
    # Extract sub-matrices
    M_m1_sub = M_m1[:, active_indices]
    M_0_sub = M_0[:, active_indices]
    M_1_sub = M_1[:, active_indices]
    M_2_sub = M_2[:, active_indices]
    
    def objective(w_vec):
        # w_vec is on the simplex (sums to 1, >= 0)
        H_m1, M = _compute_entropies(w_vec, M_m1_sub, M_0_sub, M_1_sub, M_2_sub)
        if M < 1e-9:
            return -1e9 # Penalize invalid
        return -H_m1 / M # Minimize negative ratio
        
    # Initial guess: uniform
    w0 = np.ones(k) / k
    
    # Bounds for SLSQP
    bounds = [(0, 1)] * k
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    try:
        from scipy.optimize import minimize
        # Use Nelder-Mead for robustness on small k, or SLSQP
        # SLSQP handles constraints explicitly.
        res = minimize(
            objective, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res.success:
            w_opt = res.x
            # Ensure normalization
            w_opt = w_opt / np.sum(w_opt)
            score = -res.fun
            if score < 0: score = 0.0
            return w_opt, score
        else:
            # Fallback to uniform if optimization fails
            H_m1, M = _compute_entropies(w0, M_m1_sub, M_0_sub, M_1_sub, M_2_sub)
            if M > 1e-9:
                return w0, H_m1 / M
            else:
                return None, 0.0
    except Exception:
        # Fallback
        H_m1, M = _compute_entropies(w0, M_m1_sub, M_0_sub, M_1_sub, M_2_sub)
        if M > 1e-9:
            return w0, H_m1 / M
        else:
            return None, 0.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    M_m1, M_0, M_1, M_2 = _get_pooling_matrices()
    
    best_P = np.ones((10, 10)) / 100.0
    best_score = 0.0
    
    # 1. Evaluate Uniform
    w_uni = np.ones(100) / 100.0
    H_m1, M = _compute_entropies(w_uni, M_m1, M_0, M_1, M_2)
    if M > 1e-9:
        best_score = H_m1 / M
        best_P = w_uni.reshape(10, 10)
        
    # 2. Evaluate some structured supports
    # The "Kakeya" intuition suggests lines or thin structures.
    # Let's generate candidate supports based on geometric patterns.
    
    y_idx, x_idx = np.indices((10, 10))
    
    candidates = []
    
    # Pattern A: Single diagonals (x - y = c)
    for c in range(-9, 10):
        mask = (x_idx - y_idx == c)
        idx = np.where(mask.flatten())[0]
        if len(idx) > 1:
            candidates.append(idx)
            
    # Pattern B: Single rows (y = c) -> x varies
    for c in range(10):
        mask = (y_idx == c)
        idx = np.where(mask.flatten())[0]
        candidates.append(idx)
        
    # Pattern C: Single columns (x = c) -> y varies
    for c in range(10):
        mask = (x_idx == c)
        idx = np.where(mask.flatten())[0]
        candidates.append(idx)
        
    # Pattern D: Two parallel diagonals
    # This might increase H_0, H_1, H_2 while keeping H_{-1} high?
    # Actually, H_{-1} for two diagonals is log(2) if uniform on both.
    # H_0 will be higher.
    # Let's just stick to single lines first, then try combinations.
    
    # Pattern E: Random small subsets
    rng = np.random.RandomState(42)
    for _ in range(50):
        size = rng.randint(3, 15)
        idx = rng.choice(100, size=size, replace=False)
        candidates.append(idx)
        
    # Pattern F: Structured 2D blocks
    # e.g., 2x2 blocks, 3x3 blocks
    for y0 in range(0, 10, 2):
        for x0 in range(0, 10, 2):
            mask = (y_idx >= y0) & (y_idx < y0+2) & (x_idx >= x0) & (x_idx < x0+2)
            idx = np.where(mask.flatten())[0]
            candidates.append(idx)

    # Deduplicate candidates by frozenset
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = frozenset(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
            
    # Evaluate each candidate support
    # Sort by size? Smaller supports might be faster to optimize, 
    # but larger supports might have higher entropy.
    # Let's iterate through all.
    
    # Time budget check
    if time.time() > deadline - 60:
        return best_P
        
    # We can parallelize this? No, standard lib. 
    # But each optimization is fast for k < 20.
    
    for i, active_indices in enumerate(unique_candidates):
        if time.time() > deadline - 10:
            break
            
        w_opt, score = _solve_weights_for_support(active_indices, M_m1, M_0, M_1, M_2)
        
        if w_opt is not None and score > best_score:
            best_score = score
            # Construct full P
            P_new = np.zeros(100)
            P_new[active_indices] = w_opt
            best_P = P_new.reshape(10, 10)
            
    # 3. Local Search / Refinement
    # Try adding/removing cells to the best support found.
    # Or just run a few random restarts of the optimization on the full 100D 
    # starting from the best found P?
    
    # The current best_P might have a sparse support.
    # Let's try to refine the support by checking neighbors.
    
    # Identify current active indices
    active_mask = best_P.flatten() > 1e-12
    current_active = np.where(active_mask)[0]
    
    # Try adding one random inactive cell to the support and re-optimizing
    inactive = np.where(~active_mask)[0]
    
    if len(inactive) > 0 and time.time() < deadline - 30:
        # Sample a few additions
        for _ in range(20):
            if time.time() > deadline - 10: break
            add_idx = rng.choice(inactive)
            new_active = np.union1d(current_active, [add_idx])
            w_opt, score = _solve_weights_for_support(new_active, M_m1, M_0, M_1, M_2)
            if w_opt is not None and score > best_score:
                best_score = score
                P_new = np.zeros(100)
                P_new[new_active] = w_opt
                best_P = P_new.reshape(10, 10)
                current_active = new_active
                inactive = np.where(best_P.flatten() <= 1e-12)[0]
                
    # Try removing one active cell (if support is large)
    if len(current_active) > 2 and time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 10: break
            rem_idx = rng.choice(current_active)
            new_active = np.setdiff1d(current_active, [rem_idx])
            if len(new_active) < 2: continue
            w_opt, score = _solve_weights_for_support(new_active, M_m1, M_0, M_1, M_2)
            if w_opt is not None and score > best_score:
                best_score = score
                P_new = np.zeros(100)
                P_new[new_active] = w_opt
                best_P = P_new.reshape(10, 10)
                current_active = new_active
                inactive = np.where(best_P.flatten() <= 1e-12)[0]

    # Final validation
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / total
        
    return best_P

# EVOLVE-BLOCK-END
