"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Lagrangian Dual Approximation with Greedy Refinement.
    
    1. Problem Framing:
       The multiple-constraint knapsack problem can be approximated via its 
       Lagrangian relaxation. We seek multipliers (lambda) for each constraint 
       such that the reduced cost (prize - sum(lambda * weight)) provides a 
       good ranking of items. Items with high positive reduced cost are 
       "profitable" after paying the shadow price of the resources they consume.
       
    2. Method:
       - Initialize lambda to 0.
       - Iteratively:
         a. Compute reduced costs: rc_i = prize_i - sum_c(lambda_c * weight_ic)
         b. Select items with rc_i > 0 in descending order of rc_i, checking 
            feasibility. Let S be the selected set.
         c. Update lambda_c based on constraint slack:
            If constraint c is tight (sum_{i in S} w_ic >= capacity), increase lambda_c.
            If constraint c has slack, decrease lambda_c.
         d. Use a subgradient step: lambda_c <- max(0, lambda_c + step * (cap_c - used_c))
            where step is a decaying learning rate.
       - The final heuristic is based on the reduced costs from the last iteration, 
         which balances prize against the opportunity cost of consuming capacity.
       
    3. Heuristic Construction:
       - Score_i = max(0, rc_i) + epsilon
       - This ensures items that are "net profitable" under the estimated shadow 
         prices get high scores. Items that are "net unprofitable" get low scores.
       - This directly guides the ACO to pick items that fit well within the 
         global capacity constraints, avoiding local greedy traps.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 10.0
    
    # Lagrangian Dual Approximation
    # Initialize multipliers
    lambda_c = np.zeros(n_constraints)
    
    # Number of iterations
    n_iters = 50
    
    # Step size schedule
    # Start with a reasonable step, decay over time
    step0 = 1.0
    
    # Store the best reduced costs seen
    best_rc = np.zeros(n_items)
    best_value = -1.0
    
    for it in range(n_iters):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Compute reduced costs: rc_i = prize_i - sum_c(lambda_c * weight_ic)
        # weights shape (n_constraints, n_items)
        # lambda_c shape (n_constraints,)
        # sum_c(lambda_c * weight_ic) = lambda_c @ weights
        shadow_cost = lambda_c @ weights  # shape (n_items,)
        rc = prizes - shadow_cost  # shape (n_items,)
        
        # Select items with rc > 0 in descending order of rc
        # Sort indices by rc descending
        order = np.argsort(-rc)
        
        # Greedy selection
        rem_cap = np.full(n_constraints, capacity)
        selected = np.zeros(n_items, dtype=bool)
        total_prize = 0.0
        
        for i in order:
            if rc[i] <= 0:
                break
            # Check if item i fits
            if np.all(weights[:, i] <= rem_cap):
                selected[i] = True
                rem_cap -= weights[:, i]
                total_prize += prizes[i]
        
        # Update best
        if total_prize > best_value:
            best_value = total_prize
            best_rc = rc.copy()
        
        # Update lambda_c
        # Used capacity per constraint
        used_cap = weights @ selected  # shape (n_constraints,)
        slack = capacity - used_cap  # positive if slack, negative if over (shouldn't happen)
        
        # Step size decays
        step = step0 / (1.0 + it * 0.1)
        
        # Subgradient update: lambda_c <- max(0, lambda_c + step * slack_c)
        # If slack > 0 (constraint not tight), increase lambda to penalize more.
        # If slack < 0 (shouldn't happen), decrease lambda.
        # Actually, standard subgradient for maximization with <= constraints:
        # We want to maximize sum(prize) s.t. sum(w) <= cap.
        # Lagrangian L = sum(prize) + sum(lambda * (cap - sum(w)))
        # Subgradient wrt lambda_c is (cap_c - sum_{i in S} w_ic) = slack_c
        # So lambda_c <- max(0, lambda_c + step * slack_c)
        
        lambda_c = np.maximum(0, lambda_c + step * slack)
    
    # Final heuristic based on best reduced costs
    # Ensure positive
    scores = best_rc.copy()
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
