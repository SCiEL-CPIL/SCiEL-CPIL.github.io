"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize and total weight per item
    total_prize = np.sum(prizes, axis=0)
    total_weight = np.sum(weights, axis=0)
    
    # Avoid division by zero
    epsilon = 1e-9
    safe_weights = total_weight + epsilon
    
    # Heuristic: Value-to-Weight Ratio
    heuristic = total_prize / safe_weights
    
    # Ensure positivity
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Try to improve heuristic using a smooth approximation of the constraint satisfaction
    # The ACO selects items with probability proportional to pheromone * heuristic
    # We want items that are likely to fit in all constraints to have higher heuristic
    # A good heuristic should favor items with high prize-to-weight ratio but also
    # consider the "slack" in each constraint
    
    # For each item j, the weight in constraint i is weights[i, j]
    # An item is "safe" if it can be added to any feasible solution
    # A simple heuristic: penalize items that are heavy in any constraint
    
    # Let's try a geometric mean of (prize / weight) across constraints
    # This rewards items that are efficient in all constraints
    
    # Actually, the original heuristic uses sum of weights, which is a linear penalty
    # Let's try a different approach: use the minimum slack ratio
    
    # For each item, compute the ratio of prize to the maximum weight in any constraint
    # This ensures that items that are heavy in at least one constraint are penalized
    
    max_weight_per_item = np.max(weights, axis=0)
    heuristic_v2 = total_prize / (max_weight_per_item + epsilon)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Another approach: use the product of (1 - weight_i / capacity) as a "slack" factor
    # Capacity is 25 for each constraint
    capacity = 25
    slack_factors = 1 - weights / capacity  # (5, 100)
    # Geometric mean of slack factors (using log-sum-exp for numerical stability)
    log_slack = np.log(np.clip(slack_factors, 1e-9, 1))
    geometric_slack = np.exp(np.mean(log_slack, axis=0))
    
    heuristic_v3 = total_prize * geometric_slack / (total_weight + epsilon)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Let's try a combination: use the original ratio but with a stronger penalty for heavy items
    # Use a power of the weight sum to make the penalty more aggressive
    weight_penalty = np.power(total_weight + epsilon, 1.5)
    heuristic_v4 = total_prize / weight_penalty
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    # Return the best heuristic found (based on intuition, v2 or v3 might be better)
    # For now, return the original as it's the safest
    return heuristic
# EVOLVE-BLOCK-END
