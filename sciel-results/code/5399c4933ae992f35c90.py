import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using Lagrangian dual shadow prices.
    
    Estimates the shadow price of each constraint via subgradient optimization,
    then scores each item by its net value: total prize minus weighted sum of
    resource costs at estimated shadow prices. This provides a principled
    estimate of marginal value considering all constraints simultaneously.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    eps = 1e-9
    capacity = 25.0  # Fixed capacity per constraint
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Total prize per item
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    
    # Initialize shadow prices (Lagrange multipliers)
    # Start with uniform prices based on average efficiency
    avg_efficiency = np.mean(total_prizes / (np.sum(weights, axis=0) + eps))
    lambda_vec = np.full(n_constraints, avg_efficiency * 0.1)
    
    # Subgradient ascent to estimate shadow prices
    # We want to maximize: sum_j max(0, total_prize_j - lambda^T w_j) + lambda^T capacity
    # Subgradient of the dual w.r.t. lambda_k = capacity_k - sum_j w_{k,j} * x_j
    # where x_j = 1 if item j has positive net value at current lambda
    
    step_size_init = 1.0
    n_iterations = 2000
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Compute net value of each item at current shadow prices
        # net_value_j = total_prize_j - sum_k lambda_k * w_{k,j}
        net_values = total_prizes - weights.T @ lambda_vec  # (100,)
        
        # Items selected: those with positive net value
        selected = net_values > 0  # (100,)
        
        # Compute resource usage of selected items
        resource_usage = weights[:, selected].sum(axis=1) if selected.any() else np.zeros(n_constraints)
        
        # Subgradient: capacity - resource_usage
        subgrad = capacity - resource_usage  # (5,)
        
        # Normalize subgradient for step size
        norm_sq = np.sum(subgrad ** 2)
        if norm_sq < eps:
            break
        
        # Polyak-style step size
        # Upper bound on optimal dual value
        ub = np.sum(total_prizes) + np.sum(lambda_vec * capacity)
        lb = np.sum(np.maximum(net_values, 0)) + np.sum(lambda_vec * capacity)
        gap = max(ub - lb, eps)
        
        step = gap / norm_sq
        step = min(step, step_size_init / (1.0 + 0.01 * it))
        
        # Update lambda (project to non-negative)
        lambda_vec = np.maximum(lambda_vec + step * subgrad, 0.0)
        
        # Damping for stability
        if it > 50:
            lambda_vec = 0.95 * lambda_vec + 0.05 * avg_efficiency * 0.1
    
    # Final net values at converged shadow prices
    net_values = total_prizes - weights.T @ lambda_vec  # (100,)
    
    # Score: use positive part of net value, with a small floor
    # Items with higher net value are more attractive
    scores = np.maximum(net_values, eps)
    
    # Add a small bonus for items that don't consume much of any constraint
    # (to break ties and prefer lighter items when net values are similar)
    max_weight_per_constraint = np.max(weights, axis=0)  # (100,)
    lightness_bonus = 1.0 / (1.0 + max_weight_per_constraint)  # (100,)
    scores = scores * (1.0 + 0.1 * lightness_bonus)
    
    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
