import numpy as np
import time
from scipy.optimize import linprog

def _run_aco_simulation(prizes, weights, heuristic, n_ants=20, n_iterations=30, capacity=25, seed=42):
    """
    Run a simplified ACO simulation to evaluate how well a heuristic performs.
    Returns the best total prize found.
    """
    num_constraints, n_items = prizes.shape
    rng = np.random.default_rng(seed)
    
    # Pheromone starts at 1
    pheromone = np.ones(n_items, dtype=np.float64)
    
    # Ensure heuristic is positive
    heuristic_safe = np.maximum(heuristic, 1e-9)
    
    best_overall = 0.0
    
    for iteration in range(n_iterations):
        iteration_best = 0.0
        
        for ant in range(n_ants):
            # Track remaining capacity per constraint
            remaining = np.full(num_constraints, capacity, dtype=np.float64)
            total_prize = 0.0
            selected = np.zeros(n_items, dtype=bool)
            
            while True:
                # Find items that still fit in all constraints
                fits = np.all(weights <= remaining[:, None], axis=0)
                fits[selected] = False
                
                if not np.any(fits):
                    break
                
                # Compute selection probabilities
                probs = pheromone * heuristic_safe
                probs[~fits] = 0.0
                
                prob_sum = probs.sum()
                if prob_sum <= 0:
                    break
                
                probs = probs / prob_sum
                
                # Select an item
                item_idx = rng.choice(n_items, p=probs)
                
                # Add it
                selected[item_idx] = True
                total_prize += prizes[:, item_idx].sum()
                remaining -= weights[:, item_idx]
            
        if total_prize > iteration_best:
            iteration_best = total_prize
        
        if total_prize > best_overall:
            best_overall = total_prize
        
        pheromone = pheromone * 0.9 + iteration_best / 10.0
    
    return best_overall

def _compute_lp_duals(prizes, weights, capacity=25):
    """
    Solve the LP relaxation of the knapsack problem to get dual prices (shadow prices).
    Maximize sum_j (sum_c prizes[c,j]) * x_j
    subject to:
      sum_j weights[c,j] * x_j <= capacity for each constraint c
      0 <= x_j <= 1
    
    Returns dual variables (shadow prices) for the capacity constraints.
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    
    # LP: max c^T x s.t. A x <= b, 0 <= x <= 1
    # c = total_prize
    # A = weights (num_constraints x n_items)
    # b = capacity for each constraint
    
    # scipy linprog minimizes, so we negate c
    c = -total_prize
    
    # Inequality constraints: A_ub x <= b_ub
    A_ub = weights
    b_ub = np.full(num_constraints, capacity)
    
    # Bounds: 0 <= x_j <= 1
    bounds = [(0, 1)] * n_items
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:  # Optimal
            # Dual variables are in res.ineqlin.marginals (negative of shadow prices)
            # For max problem with <= constraints, shadow prices are negative of marginals
            # Actually, for linprog min c^T x s.t. A_ub x <= b_ub:
            # The dual of the constraint A_ub x <= b_ub is y >= 0
            # The marginal is d(optimal_value)/d(b_ub), which for min is the shadow price
            # For our max problem, we want the shadow price which is how much the objective
            # increases per unit increase in capacity.
            
            # res.ineqlin.marginals gives d(optimal_min_value)/d(b_ub)
            # Since we're minimizing -total_prize, the marginal is negative of the shadow price
            # for the max problem.
            
            duals = -res.ineqlin.marginals
            # Ensure duals are non-negative (they should be for <= constraints in max)
            duals = np.maximum(duals, 0)
            return duals
        else:
            # Fallback: use average weight as a rough dual estimate
            return np.full(num_constraints, 1.0 / (capacity + 1e-9))
    except Exception:
        # Fallback: use average weight as a rough dual estimate
        return np.full(num_constraints, 1.0 / (capacity + 1e-9))

def _compute_reduced_cost_heuristic(prizes, weights, duals, capacity=25):
    """
    Compute reduced cost heuristic: prize - sum_c dual_c * weight_c
    This identifies items that are most efficient relative to the LP optimum.
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    
    # Reduced cost: total_prize - sum_c dual_c * weights[c, j]
    # For each item j: reduced_cost[j] = total_prize[j] - sum_c duals[c] * weights[c, j]
    reduced_costs = total_prize - duals @ weights
    
    # The heuristic should favor items with high reduced costs
    # But reduced costs can be negative, so we need to ensure positivity
    # Shift so that the minimum is a small positive value
    min_rc = np.min(reduced_costs)
    if min_rc < 0:
        reduced_costs = reduced_costs - min_rc + 1e-9
    
    # Scale to reasonable range
    max_rc = np.max(reduced_costs)
    if max_rc > min_rc:
        reduced_costs = (reduced_costs - min_rc) / (max_rc - min_rc) + 1e-9
    
    return reduced_costs

def _compute_pareto_heuristic(prizes, weights):
    """
    Compute a Pareto dominance-based heuristic.
    Items that are not dominated in prize-weight space get higher scores.
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    sum_weights = weights.sum(axis=0)
    
    # An item j dominates item i if:
    # total_prize[j] >= total_prize[i] and sum_weights[j] <= sum_weights[i]
    # and at least one is strict
    
    # For simplicity, score items by their "Pareto rank" - lower rank is better
    # Non-dominated items get rank 1, then remove them and repeat
    
    ranks = np.zeros(n_items, dtype=np.float64)
    remaining = np.ones(n_items, dtype=bool)
    current_rank = 1
    
    while np.any(remaining):
        remaining_indices = np.where(remaining)[0]
        if len(remaining_indices) == 0:
            break
        
        # Find non-dominated items among remaining
        non_dominated = np.zeros(len(remaining_indices), dtype=bool)
        
        for i_idx, i in enumerate(remaining_indices):
            dominated = False
            for j_idx, j in enumerate(remaining_indices):
                if i == j:
                    continue
                # Check if j dominates i
                if (total_prize[j] >= total_prize[i] and 
                    sum_weights[j] <= sum_weights[i] and 
                    (total_prize[j] > total_prize[i] or sum_weights[j] < sum_weights[i])):
                    dominated = True
                    break
            non_dominated[i_idx] = not dominated
        
        # Assign rank to non-dominated items
        for i_idx, i in enumerate(remaining_indices):
            if non_dominated[i_idx]:
                ranks[i] = current_rank
                remaining[i] = False
        
        if not np.any(non_dominated):
            # No progress, break to avoid infinite loop
            break
        
        current_rank += 1
    
    # Heuristic: lower rank is better, so use 1/rank
    # For items that didn't get a rank, use a large rank
    ranks = np.maximum(ranks, 1e-9)
    heuristic = 1.0 / ranks
    
    # Normalize
    max_h = np.max(heuristic)
    if max_h > 0:
        heuristic = heuristic / max_h
    
    return np.maximum(heuristic, 1e-9)

def _compute_combined_heuristic(prizes, weights, deadline=None):
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 280
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    num_constraints, n_items = prizes.shape
    
    # 1. Compute LP duals and reduced cost heuristic
    duals = _compute_lp_duals(prizes, weights)
    rc_heuristic = _compute_reduced_cost_heuristic(prizes, weights, duals)
    
    # 2. Compute Pareto-based heuristic
    pareto_heuristic = _compute_pareto_heuristic(prizes, weights)
    
    # 3. Combine: weighted geometric mean or product
    # The reduced cost heuristic captures efficiency relative to LP optimum
    # The Pareto heuristic captures non-dominance in prize-weight space
    # Combine multiplicatively to emphasize items that are good in both aspects
    
    # Normalize both to [0, 1] range for consistent scaling
    rc_norm = (rc_heuristic - np.min(rc_heuristic)) / (np.max(rc_heuristic) - np.min(rc_heuristic) + 1e-9)
    pareto_norm = (pareto_heuristic - np.min(pareto_heuristic)) / (np.max(pareto_heuristic) - np.min(pareto_heuristic) + 1e-9)
    
    # Geometric mean with slight weighting toward reduced cost
    # h = rc_norm^0.6 * pareto_norm^0.4
    combined = np.power(rc_norm + 1e-9, 0.6) * np.power(pareto_norm + 1e-9, 0.4)
    
    # Ensure positivity
    combined = np.maximum(combined, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        pass  # We've already computed, just return
    
    return combined

def heuristics(prizes, weights):
    start_time = time.time()
    try:
        result = _compute_combined_heuristic(prizes, weights, start_time + 280)
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception:
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
        return np.ones(n_items, dtype=np.float64)
