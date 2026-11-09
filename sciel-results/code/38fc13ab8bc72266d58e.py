import numpy as np
from scipy.optimize import linprog

def heuristics(prizes, weights):
    # Ensure prizes is 1D
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    
    n_constraints = weights.shape[0]
    n_items = weights.shape[1]
    capacity = 25.0
    
    # Solve the dual LP: min sum(capacity * y_c) s.t. sum(y_c * w_ci) >= p_i for all i, y_c >= 0
    # linprog solves: min c^T x s.t. A_ub @ x <= b_ub, A_eq @ x = b_eq, bounds
    # We need: min 25 * sum(y_c) s.t. -sum(w_ci * y_c) <= -p_i for all i, y_c >= 0
    c_obj = np.full(n_constraints, capacity)
    A_ub = -weights  # shape (n_items, n_constraints), each row: -w_ci for item i
    b_ub = -prizes    # shape (n_items,)
    bounds = [(0, None)] * n_constraints
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            dual_prices = res.x  # shape (n_constraints,)
        else:
            # Fallback: use sum of weights as before
            dual_prices = np.ones(n_constraints)
    except Exception:
        dual_prices = np.ones(n_constraints)
    
    # Compute reduced cost for each item: prize - sum(dual_price[c] * weight[c][i])
    # weights shape: (n_constraints, n_items)
    # reduced_cost[i] = prizes[i] - sum_c(dual_prices[c] * weights[c][i])
    weighted_sum = weights.T @ dual_prices  # shape (n_items,)
    reduced_cost = prizes - weighted_sum
    
    # The heuristic should be larger = more attractive
    # Items with higher reduced cost are better relative to the LP optimum
    # Use a smooth positive transformation to ensure positivity
    # Add a baseline so all values are positive, then use exp for smooth ranking
    # Or simply: shift and scale to be positive
    # We want items with high reduced cost to have high heuristic
    # Use: heuristic = exp(reduced_cost / scale) or (reduced_cost - min) + epsilon
    
    # Use a robust positive mapping
    min_rc = np.min(reduced_cost)
    max_rc = np.max(reduced_cost)
    range_rc = max_rc - min_rc + 1e-9
    
    # Normalize to [0, 1] then shift to be positive
    normalized = (reduced_cost - min_rc) / range_rc
    heuristic = normalized + 0.1  # shift so minimum is 0.1, not zero
    
    # Ensure positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
