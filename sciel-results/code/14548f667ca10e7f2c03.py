import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns an array of one attractiveness score per item.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    
    # Aggregate prizes to a single value per item
    # Assuming total prize is the sum of all components
    p = prizes.sum(axis=0)
    
    # Generate candidate heuristics
    candidates = []
    
    # Basic aggregations
    w_sum = weights.sum(axis=0)
    w_max = weights.max(axis=0)
    w_min = weights.min(axis=0)
    w_mean = weights.mean(axis=0)
    w_sq_sum = np.sum(weights**2, axis=0)
    w_l2 = np.linalg.norm(weights, axis=0)
    
    # Combinations
    w_sum_max = w_sum + w_max
    w_sum_min = w_sum + w_min
    w_max_min = w_max + w_min
    
    # Define cost functions
    cost_funcs = [
        ("sum", w_sum),
        ("max", w_max),
        ("min", w_min),
        ("mean", w_mean),
        ("sq_sum", w_sq_sum),
        ("l2", w_l2),
        ("sum_max", w_sum_max),
        ("sum_min", w_sum_min),
        ("max_min", w_max_min),
        ("sum_sq_max", w_sq_sum + w_max),
        ("sum_sq_min", w_sq_sum + w_min),
        ("inv_min", 1.0 / (w_min + 1e-9)), # Inverse min weight (high if min weight is small)
        ("inv_max", 1.0 / (w_max + 1e-9)),
        ("inv_mean", 1.0 / (w_mean + 1e-9)),
    ]
    
    best_heuristic = None
    best_sim_score = -1.0
    
    for name, cost in cost_funcs:
        # Check time
        if time.time() > deadline:
            break
            
        # Compute heuristic: p / cost
        # If cost is an inverse metric, we might want p * cost?
        # For "inv_*", the cost is small when weight is large.
        # We want high heuristic for good items.
        # Good items: High P, Low Weight.
        # If cost = 1/w, then p/cost = p*w. This rewards heavy items. Bad.
        # So for "inv_*", we should use p * cost?
        # Or just treat "cost" as the denominator.
        # Let's stick to p / cost for all, but define cost appropriately.
        # For inv_min, cost = 1/w_min. p/cost = p*w_min. Rewards heavy.
        # So for inv_*, we should use p * cost?
        # Let's just use p / cost for the standard ones, and p * cost for the inverse ones?
        # No, let's just define the heuristic directly.
        
        if name.startswith("inv_"):
            h = p * cost
        else:
            h = p / (cost + 1e-9)
            
        # Ensure positive and finite
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Simulate ACO
        sim_score = simulate_aco(prizes, weights, h, p)
        
        if sim_score > best_sim_score:
            best_sim_score = sim_score
            best_heuristic = h
    
    # If no candidate was evaluated or all failed, return a default
    if best_heuristic is None:
        best_heuristic = p / (w_sum + 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
        
    return best_heuristic

def simulate_aco(prizes, weights, heuristic, p):
    """
    Runs a fast ACO simulation to estimate the quality of the heuristic.
    """
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    n_ants = 20
    n_iters = 30 # Match real iterations
    
    tau = np.ones(n_items)
    best_total_prize = 0.0
    
    # Pre-compute weights as list of columns for faster access?
    # weights is (5, 100). weights[:, i] is a column.
    
    for it in range(n_iters):
        best_iter_prize = 0.0
        ant_prizes = []
        
        for ant in range(n_ants):
            remaining_cap = np.full(n_constraints, capacity)
            selected = np.zeros(n_items, dtype=bool)
            total_prize = 0.0
            
            # Selection loop
            # We can optimize by maintaining a list of available items
            # But 100 items is small, so full scan is fine.
            
            while True:
                # Check fit: weights[:, i] <= remaining_cap
                # Vectorized:
                # fits = np.all(weights <= remaining_cap[:, None], axis=0)
                # & ~selected
                fits = np.all(weights <= remaining_cap[:, None], axis=0) & ~selected
                
                if not np.any(fits):
                    break
                
                candidates_idx = np.where(fits)[0]
                
                # Calculate probabilities
                # prob_i = tau_i * h_i
                weights_prob = tau[candidates_idx] * heuristic[candidates_idx]
                weights_prob = np.maximum(weights_prob, 1e-9)
                sum_w = np.sum(weights_prob)
                if sum_w < 1e-12:
                    # Fallback to uniform
                    probs = np.ones(len(candidates_idx)) / len(candidates_idx)
                else:
                    probs = weights_prob / sum_w
                
                # Pick one
                idx = np.random.choice(candidates_idx, p=probs)
                
                # Update state
                selected[idx] = True
                remaining_cap -= weights[:, idx]
                total_prize += p[idx] # Use aggregated prize
                
        ant_prizes.append(total_prize)
        
        best_iter_prize = max(ant_prizes)
        if best_iter_prize > best_total_prize:
            best_total_prize = best_iter_prize
        
        # Update pheromone
        tau *= 0.9
        tau += best_iter_prize / 10.0
        
    return best_total_prize
