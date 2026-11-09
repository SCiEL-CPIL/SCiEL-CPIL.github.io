import numpy as np
import time

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
        
        if iteration_best > best_overall:
            best_overall = iteration_best
        
        pheromone = pheromone * 0.9 + iteration_best / 10.0
    
    return best_overall

def _estimate_dual_prices(prizes, weights, capacity=25, max_iter=50, lr=0.5):
    """
    Estimate constraint dual prices via subgradient method on the Lagrangian dual.
    The Lagrangian for the knapsack is:
    L(lambda) = max_{x in {0,1}^n} sum_j (prize_j - sum_c lambda_c * w_jc) x_j + sum_c lambda_c * C_c
    
    The subgradient of the dual function g(lambda) = L(lambda) is:
    g'(lambda) = sum_c C_c - sum_j w_jc * x_j*(lambda)
    
    where x*(lambda) is the optimal solution to the Lagrangian relaxation.
    Since the Lagrangian relaxation decouples into individual items, x_j* = 1 if 
    (prize_j - sum_c lambda_c * w_jc) > 0, else 0.
    
    We want to find lambda >= 0 such that the selected items satisfy the constraints
    approximately (sum_j w_jc * x_j* <= C_c).
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    
    # Initialize dual prices to zero
    lambda_vec = np.zeros(num_constraints)
    
    # Precompute for efficiency
    # For each item j, the reduced cost is prize_j - lambda @ weights[:, j]
    # We select items with positive reduced cost.
    
    for _ in range(max_iter):
        # Compute reduced costs
        # reduced_cost[j] = total_prize[j] - sum_c lambda[c] * weights[c, j]
        reduced_costs = total_prize - weights.T @ lambda_vec
        
        # Select items with positive reduced cost
        selected = reduced_costs > 0
        
        # Compute usage per constraint
        usage = weights[:, selected].sum(axis=1)
        
        # Subgradient: capacity - usage
        # If usage < capacity, we can increase lambda (making items less attractive)
        # Wait, standard subgradient update for minimization of dual:
        # lambda_new = max(0, lambda_old + alpha * (usage - capacity))
        # We want to minimize g(lambda) = max_x (profit - lambda@w) + lambda@C
        # The subgradient is C - usage.
        # To minimize, we move in direction of negative subgradient: - (C - usage) = usage - C.
        # So lambda_new = max(0, lambda_old + alpha * (usage - C))
        
        # If usage < C, usage - C is negative, lambda decreases (or stays 0).
        # If usage > C, usage - C is positive, lambda increases.
        # This makes sense: if we are over capacity, increase the price of that resource.
        
        subgrad = usage - capacity
        
        # Update lambda
        # Adaptive step size
        # Norm of subgradient
        sg_norm = np.linalg.norm(subgrad)
        if sg_norm < 1e-6:
            break
            
        # Simple fixed step size or adaptive
        # Let's use a step size that scales with the gap
        # alpha = 2 * (UB - LB) / ||subgrad||^2 is standard, but we don't have LB easily.
        # Let's use a simple decay or fixed step.
        # Given values are in [0,1], capacity 25, prizes sum up to ~2.5.
        # A step size of 0.1 to 0.5 seems reasonable.
        alpha = 0.5
        
        lambda_vec = np.maximum(0, lambda_vec + alpha * subgrad)
        
        # Check convergence: if usage is close to capacity for all constraints
        # or if lambda stops changing significantly
        if np.max(np.abs(subgrad)) < 0.1:
            break

    return lambda_vec

def _compute_heuristics(prizes, weights, deadline=None):
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 290
        
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    num_constraints, n_items = prizes.shape
    
    # 1. Estimate dual prices using Lagrangian relaxation
    dual_prices = _estimate_dual_prices(prizes, weights, capacity=25, max_iter=100, lr=0.5)
    
    # 2. Compute reduced costs
    # Reduced cost = Prize - Dual Price @ Weight
    # Items with high positive reduced cost are "valuable" relative to their resource consumption
    total_prize = prizes.sum(axis=0)
    reduced_costs = total_prize - weights.T @ dual_prices
    
    # 3. Combine with a base efficiency heuristic to ensure robustness
    # If dual prices are all zero (e.g., if no item fits or relaxation is trivial),
    # fall back to prize/weight ratio.
    
    sum_weights = weights.sum(axis=0)
    max_weights = weights.max(axis=0)
    eps = 1e-9
    
    # Base heuristic: prize / (sum of weights)
    base_heuristic = total_prize / (sum_weights + eps)
    
    # Normalize base heuristic to [0, 1]
    def norm(x):
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-6:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)
        
    norm_base = norm(base_heuristic)
    
    # Normalize reduced costs to [0, 1]
    # Note: reduced costs can be negative. We want to map them such that higher is better.
    # If all are negative, the best is still the least negative.
    rc_min = reduced_costs.min()
    rc_max = reduced_costs.max()
    if rc_max - rc_min < 1e-6:
        norm_rc = np.ones_like(reduced_costs)
    else:
        norm_rc = (reduced_costs - rc_min) / (rc_max - rc_min)
    
    # Combine: Weighted sum of normalized reduced cost and base heuristic
    # Reduced cost is theoretically stronger, but base heuristic provides a stable floor.
    # Let's try 0.7 * norm_rc + 0.3 * norm_base
    alpha = 0.7
    heuristic = alpha * norm_rc + (1 - alpha) * norm_base
    
    # Ensure positivity
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 4. Optional: Refine with a quick local search or ACO evaluation if time permits
    # Given the budget, a full ACO eval is expensive. Let's stick to the analytical heuristic
    # unless we have significant time left.
    
    if time.time() < deadline - 30:
        # We have time to test a few variations or run a quick ACO to validate
        # However, running ACO takes ~1-2s per seed. 3 seeds = 6s.
        # Let's try to see if a simple scaling of the heuristic improves things.
        # Actually, the ACO simulation is deterministic given the heuristic and seed.
        # We can test 2-3 seeds to ensure stability.
        
        # Let's just return the analytical heuristic. It's fast and grounded in theory.
        # The "Pareto dominance" idea from the inspiration might be better implemented
        # as a filter, but reduced costs already handle the trade-off.
        
        pass
    
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic

def heuristics(prizes, weights):
    start_time = time.time()
    try:
        result = _compute_heuristics(prizes, weights, start_time + 290)
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
