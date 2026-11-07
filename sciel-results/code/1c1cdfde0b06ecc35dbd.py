import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # prizes: (5, 100), weights: (5, 100)
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    epsilon = 1e-9
    
    # 1. Calculate Efficiency Ratios R_kj
    # Avoid division by zero. If weight is 0, we treat it as 1.0 to avoid inf.
    # If prize is 0, ratio is 0.
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    ratios = prizes / safe_weights
    
    # Ensure non-negative (prizes are uniform [0,1] so they are non-negative, but just in case)
    ratios = np.maximum(ratios, 0.0)
    
    # 2. Harmonic Mean of Ratios
    # H_j = n / sum(1/R_kj)
    # If R_kj is 0, 1/R is inf. Harmonic Mean becomes 0.
    # This is the desired behavior: if an item has 0 efficiency in any constraint, its harmonic mean is 0.
    
    # To compute 1/R safely:
    # We want inv_ratios[k, j] = 1 / ratios[k, j]
    # If ratios[k, j] == 0, inv_ratios should be inf (or a large number) to make H_j 0.
    # Using np.where: if ratio > 0, 1/ratio, else large number.
    large_num = 1e18
    inv_ratios = np.where(ratios > epsilon, 1.0 / ratios, large_num)
    
    sum_inv = np.sum(inv_ratios, axis=0) # Shape (100,)
    
    # H_j = 5 / sum_inv
    # If sum_inv is large (due to zeros), H_j is small.
    H = 5.0 / np.where(sum_inv > epsilon, sum_inv, large_num)
    
    # 3. Soft-Min of Ratios
    # SoftMin(x, beta) = beta * log(sum(exp(-x/beta)))
    # We want to approximate min(x).
    # If beta is small, it's close to min.
    # If x is 0, exp(0) = 1.
    # If x is large, exp(-large) is 0.
    # So the sum is dominated by the smallest x.
    beta = 0.1
    # To avoid overflow in exp, we can shift by min, but since we want min, 
    # standard softmin formula works if x are not huge.
    # Ratios are p/w. p in [0,1], w in [0,1]. Max ratio is 1/epsilon ~ 1e9.
    # exp(-1e9/0.1) is 0.
    # exp(0) is 1.
    # This is numerically stable for the "min" part.
    
    # However, if all ratios are very large, the sum of exps might be small?
    # No, if min is large, say 1000. exp(-10000) is 0.
    # Wait, if min is 1000, the term for the min is exp(-10000) which is 0.
    # This means SoftMin returns 0 for large positive numbers?
    # Let's check the formula.
    # SoftMin is usually defined for minimizing cost.
    # If we want to approximate min(x), and x > 0:
    # min(x) = - SoftMax(-x).
    # SoftMax(y) = log(sum(exp(y))).
    # So min(x) = - log(sum(exp(-x))).
    # My formula: beta * log(sum(exp(-x/beta))).
    # As beta -> 0, this is log(exp(-min(x)/beta)) = -min(x)/beta.
    # So beta * (-min(x)/beta) = -min(x).
    # This gives -min(x). We want +min(x).
    # So we should use: -beta * log(sum(exp(-x/beta)))?
    # Let's re-derive.
    # We want a function f(x) ~ min(x).
    # Let g(x) = log(sum(exp(x/beta))). As beta->0, g(x) ~ max(x)/beta.
    # So beta*g(x) ~ max(x).
    # Therefore, -beta * log(sum(exp(-x/beta))) ~ min(x).
    
    # Let's use:
    # S_j = -beta * log(sum(exp(-R_kj / beta)))
    
    # Numerical stability:
    # If R is very large, exp(-R/beta) is 0.
    # If R is 0, exp(0) = 1.
    # Sum is at least 1 if any R is 0.
    # If all R are large, sum is ~ 0. log(0) is -inf.
    # -beta * (-inf) = inf.
    # This is correct: if all ratios are large, min is large.
    
    # But we need to handle the case where sum is 0.
    # If all R are huge, sum(exp(-R/beta)) underflows to 0.
    # We can add a small constant or use a safe log-sum-exp.
    # However, ratios are bounded by 1/epsilon.
    # Let's just use the formula and handle non-finite results.
    
    # Optimization: 
    # S = -beta * log(sum(exp(-R/beta)))
    # To avoid underflow, we can shift by min(R):
    # S = min(R) - beta * log(sum(exp(-(R - min(R))/beta)))
    # This is the standard stable SoftMin.
    
    R_min = np.min(ratios, axis=0)
    # Shifted ratios
    shifted = ratios - R_min[:, np.newaxis]
    # exp(-shifted / beta)
    # If shifted is 0, exp(0)=1.
    # If shifted is large, exp(-large)=0.
    exp_terms = np.exp(-shifted / beta)
    sum_exp = np.sum(exp_terms, axis=0)
    
    # log(sum_exp)
    # sum_exp is at least 1 (because min term is 1).
    log_sum_exp = np.log(sum_exp)
    
    S = R_min - beta * log_sum_exp
    
    # 4. Combine H and S
    # H^1.5 * S^0.5
    # If H is 0, result is 0.
    # If S is 0, result is 0.
    
    # Handle zeros/negatives in H and S before power
    H_safe = np.maximum(H, epsilon)
    S_safe = np.maximum(S, epsilon)
    
    heuristic = (H_safe ** 1.5) * (S_safe ** 0.5)
    
    # 5. Normalization
    # Normalize to mean 1
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # 6. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
