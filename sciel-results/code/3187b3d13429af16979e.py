import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 5.0

def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    cap = 25.0
    capacities = np.full(n_constraints, cap, dtype=np.float64)
    
    # 1. Local Efficiency Heuristic (Soft-min of ratios)
    ratios = prizes / weights  # shape (5, 100)
    beta = 10.0
    neg_beta_ratios = -beta * ratios
    max_neg = np.max(neg_beta_ratios, axis=0)
    lse = max_neg + np.log(np.sum(np.exp(neg_beta_ratios - max_neg), axis=0))
    soft_min_ratio = -lse / beta
    
    # 2. Lagrangian Relaxation to estimate Shadow Prices
    # This helps identify which items are "expensive" in terms of resource consumption.
    # We maximize sum(p_i x_i) s.t. sum(w_ij x_j) <= C_j
    # Dual: Min sum(C_j lambda_j) s.t. sum(w_ij lambda_j) >= p_i, lambda_j >= 0
    
    lambda_vec = np.zeros(n_constraints, dtype=np.float64)
    best_reduced_costs = None
    best_val = -1.0
    
    # Subgradient method parameters
    n_iter = 100
    # Initial step size, decaying
    gamma = 1.0
    
    for k in range(n_iter):
        if _deadline_left() < MARGIN + 2.0:
            break
            
        # Reduced costs: r_i = p_i - sum_j w_ij lambda_j
        shadow_costs = weights.T @ lambda_vec  # shape (100,)
        reduced_costs = prizes.sum(axis=0) - shadow_costs
        
        # Greedy solve of the relaxed problem (ignoring integrality, but using greedy packing)
        # Order by reduced cost
        order = np.argsort(-reduced_costs, kind='stable')
        
        rem = capacities.copy()
        selected = []
        val = 0.0
        
        for i in order:
            if np.all(weights[:, i] <= rem):
                selected.append(i)
                rem -= weights[:, i]
                val += prizes.sum(axis=0)[i]
        
        if val > best_val:
            best_val = val
            best_reduced_costs = reduced_costs.copy()
        
        # Update lambda
        # Violation: C_j - sum_{i in S} w_ij
        usage = np.zeros(n_constraints)
        if selected:
            usage = weights[:, selected].sum(axis=1)
        
        violations = capacities - usage
        
        # Step size: gamma / sqrt(k+1) is standard, but we can use a fixed small step if unstable
        # Let's use a decaying step to ensure convergence
        step = gamma / np.sqrt(k + 1.0)
        
        # Subgradient update: lambda = max(0, lambda + step * violations)
        lambda_vec = np.maximum(0.0, lambda_vec + step * violations)
        
        # Optional: If we have a lot of time, we can try to polish, but 100 iters is fast.
        
    # If the loop didn't run or failed, fallback to simple ratios
    if best_reduced_costs is None:
        best_reduced_costs = prizes.sum(axis=0)
        
    # 3. Combine Heuristics
    # The reduced costs from Lagrangian relaxation indicate the marginal value of an item
    # given the estimated scarcity of resources.
    # We want to boost items with high reduced costs.
    
    # Normalize reduced costs to be positive and comparable
    # Reduced costs can be negative if the item is "too expensive" given current lambdas.
    # We clip to 0.
    h_lag = np.maximum(best_reduced_costs, 1e-9)
    
    # Normalize h_lag to [0, 1] range roughly? Or just use it as a multiplier?
    # Let's normalize both components to have similar scales.
    
    # Normalize soft_min_ratio
    sm_min = soft_min_ratio.min()
    sm_max = soft_min_ratio.max()
    if sm_max > sm_min:
        h_ratio_norm = (soft_min_ratio - sm_min) / (sm_max - sm_min)
    else:
        h_ratio_norm = np.ones_like(soft_min_ratio)
        
    # Normalize h_lag
    lag_min = h_lag.min()
    lag_max = h_lag.max()
    if lag_max > lag_min:
        h_lag_norm = (h_lag - lag_min) / (lag_max - lag_min)
    else:
        h_lag_norm = np.ones_like(h_lag)
        
    # Combine: Geometric mean or weighted sum?
    # Geometric mean is good for balancing "local efficiency" and "global scarcity".
    # h = h_ratio_norm * h_lag_norm
    # Or a power mean to allow one to dominate?
    # Let's try a weighted geometric mean: h = (h_ratio_norm)^0.5 * (h_lag_norm)^0.5
    
    # Also consider "slack": items that are small are safer.
    max_weight = weights.max(axis=0)
    slack = 1.0 / (max_weight + 1e-9)
    sl_min = slack.min()
    sl_max = slack.max()
    if sl_max > sl_min:
        h_slack_norm = (slack - sl_min) / (sl_max - sl_min)
    else:
        h_slack_norm = np.ones_like(slack)
        
    # Final heuristic
    # Weights: Lagrangian is usually the strongest signal for MCKP.
    # Let's use: h = h_lag_norm * (0.5 + 0.5 * h_ratio_norm) * (0.5 + 0.5 * h_slack_norm)
    # This ensures lag dominates, but ratio and slack provide fine-tuning.
    
    heuristic = h_lag_norm * (0.5 + 0.5 * h_ratio_norm) * (0.5 + 0.5 * h_slack_norm)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
