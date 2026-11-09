# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _fast_ratio(P):
    """
    Fast computation of D/M using vectorized bincount-style approach.
    P[y, x] is the joint probability for x, y in 0..9.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=P.flatten(), minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=P.flatten(), minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=P.flatten(), minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=P.flatten(), minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M

def _precompute_tables():
    """Precompute index arrays for fast evaluation."""
    y_idx, x_idx = np.indices((10, 10))
    v_minus1 = (x_idx - y_idx + 9).flatten()
    v_x = x_idx.flatten()
    v_plus1 = (x_idx + y_idx).flatten()
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    
    # Mapping from flattened index to bin index
    # We need to map P.flatten() to counts arrays
    # bincount does this, but for gradient we might want explicit matrices
    # However, L-BFGS-B uses finite differences, so we just need a fast forward pass.
    
    return v_minus1, v_x, v_plus1, v_plus2

def _smoothed_ratio(log_P, temperature):
    """
    Compute the ratio D/M with a smoothed max for M.
    log_P is a 100-dim array of log-probabilities.
    """
    # Convert log_P to P
    P = np.exp(log_P - np.max(log_P))
    P_sum = P.sum()
    if P_sum < 1e-9:
        return -np.inf
    P = P / P_sum
    
    y_idx, x_idx = np.indices((10, 10))
    
    # D: H_{-1}
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=P.flatten(), minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)
    
    # M: max(H_0, H_1, H_2)
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=P.flatten(), minlength=10)
    H_0 = _entropy_from_counts(counts_x)
    
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=P.flatten(), minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)
    
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=P.flatten(), minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)
    
    entropies = np.array([H_0, H_1, H_2])
    
    # Smoothed max: log-sum-exp scaled by 1/temperature
    # max(x) ~ T * log(sum(exp(x/T))) as T->0
    # We want to minimize -Ratio = -(D / M)
    # So we want to maximize D/M.
    # Let's compute M_smooth = T * log(sum(exp(H_i/T)))
    # To avoid overflow/underflow, shift by max
    max_h = np.max(entropies)
    if max_h == 0:
        M_smooth = 0
    else:
        # If temperature is very small, this approaches max
        # If temperature is large, this approaches mean
        exp_terms = np.exp((entropies - max_h) / temperature)
        M_smooth = max_h + temperature * np.log(np.sum(exp_terms))
        
    if M_smooth < 1e-9:
        return -np.inf
        
    ratio = H_minus1 / M_smooth
    return -ratio # We minimize the negative

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    n = 10
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _fast_ratio(best_P)
    
    # Initial candidates
    # 1. Uniform
    # 2. Diagonal uniform (equal weight per diagonal)
    P_diag = np.zeros((n, n), dtype=np.float64)
    y_idx, x_idx = np.indices((n, n))
    diff = x_idx - y_idx
    for d in range(-9, 10):
        mask = (diff == d)
        num_cells = mask.sum()
        if num_cells > 0:
            P_diag[mask] = 1.0 / (19.0 * num_cells)
    P_diag = P_diag / P_diag.sum()
    score_diag = _fast_ratio(P_diag)
    if score_diag > best_score:
        best_P = P_diag.copy()
        best_score = score_diag

    # 3. Corner concentration
    P_corner = np.zeros((n, n))
    P_corner[0,0] = 1.0
    score_corner = _fast_ratio(P_corner)
    if score_corner > best_score:
        best_P = P_corner.copy()
        best_score = score_corner

    # Convert best_P to log-space for optimization
    # Add small epsilon to avoid log(0)
    log_P_init = np.log(best_P + 1e-15)
    
    # Multi-start L-BFGS-B with annealing temperature
    # We will run optimization from several starting points
    # and with several temperatures
    
    rng = np.random.RandomState(42)
    
    # Define a few structured starting points in log-space
    starts = []
    
    # Start 1: Current best
    starts.append(log_P_init)
    
    # Start 2: Uniform
    starts.append(np.full((100,), np.log(1/100)))
    
    # Start 3: Diagonal uniform
    log_P_diag = np.log(P_diag + 1e-15)
    starts.append(log_P_diag)
    
    # Start 4: Random sparse
    for _ in range(5):
        P_rand = rng.dirichlet(np.ones(100) * 0.5)
        starts.append(np.log(P_rand + 1e-15))
        
    # Start 5: Concentrated on few cells
    for _ in range(5):
        P_rand = np.zeros(100)
        idx = rng.choice(100, size=10, replace=False)
        P_rand[idx] = 1.0
        P_rand /= P_rand.sum()
        starts.append(np.log(P_rand + 1e-15))

    # Optimization parameters
    # We want to maximize Ratio.
    # L-BFGS-B minimizes. So we minimize -Ratio.
    # We use smoothed max for M to make the function differentiable.
    # Temperature controls the sharpness of the max.
    # Low T -> sharp max (closer to true objective).
    # High T -> smooth average (easier to optimize).
    
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    for start_idx, start_log_P in enumerate(starts):
        if time.time() > deadline - 30:
            break
            
        # Anneal temperature
        for T in temperatures:
            if time.time() > deadline - 10:
                break
                
            # Rescale start_log_P if it's from a previous temperature?
            # No, just use the start point.
            # However, if we found a good point at high T, use it as start for lower T.
            if start_idx == 0 and T < temperatures[0]:
                 # Use the best found so far in this start chain?
                 # For simplicity, let's just run from the initial start point for each T
                 # Or better, chain them: use the result of previous T as start for next T.
                 pass
            
            # Actually, let's do a proper annealing chain for the best starting points
            # For now, run independent optimizations for each T from the best initial P
            # This is less efficient but safer.
            
            # Let's refine the best_P found so far using the current temperature
            # Convert best_P to log space
            current_log_P = np.log(best_P + 1e-15)
            
            def objective(log_P_vec):
                return _smoothed_ratio(log_P_vec, T)
            
            try:
                res = minimize(
                    objective,
                    current_log_P,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                
                if res.success and res.fun < -best_score + 1e-6:
                    # Convert back to P
                    P_opt = np.exp(res.x - np.max(res.x))
                    P_opt = P_opt / P_opt.sum()
                    P_opt = P_opt.reshape(10, 10)
                    
                    score_opt = _fast_ratio(P_opt)
                    if score_opt > best_score:
                        best_P = P_opt
                        best_score = score_opt
                        
            except Exception:
                pass
                
            # If we improved, update the starting point for the next (lower) temperature
            # This is handled by using best_P in the next iteration of the T loop
            
    # Final local polish with random restarts if time permits
    if time.time() < deadline - 60:
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            # Random perturbation of best_P
            P_test = best_P + rng.normal(0, 0.01, size=(10,10))
            P_test[P_test < 0] = 0
            if P_test.sum() < 1e-9: continue
            P_test /= P_test.sum()
            score_test = _fast_ratio(P_test)
            if score_test > best_score:
                best_P = P_test
                best_score = score_test

    # Final cleanup
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((10,10)) / 100.0
        
    return best_P

# EVOLVE-BLOCK-END
