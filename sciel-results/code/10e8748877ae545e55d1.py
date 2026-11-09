"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: Ensure we return quickly if we are running out of time.
    # The task says each evaluation must finish within 300s, but the total
    # run is 330s. We should be fast.
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or very small costs, though cost >= 1 in this task.
    # cost = 1 + mean(x_scaled), x in [0,1], so cost in [1, 2].
    safe_cost = np.maximum(cost, 1e-6)
    
    # Strategy: Linear Optimistic Improvement per Unit Cost
    # Based on knowledge: "Replacing non-linear EI with linear regret lower bound 
    # (f_best - mu + sigma) provides a smoother, more balanced acquisition function... Measured: 1 -> 1.713."
    # This corresponds to maximizing (f_best - (mu - sigma)) / cost.
    # This is an upper bound on the expected improvement (if we assume the best case within 1 sigma).
    # It encourages exploration (high sigma) and exploitation (low mu) while penalizing cost.
    
    # We use beta=1.0 as per the successful strategy.
    beta = 1.0
    
    # Calculate the optimistic improvement
    # Note: f_best is a scalar.
    improvement = f_best - mu + beta * sigma
    
    # Ensure improvement is non-negative? 
    # If mu > f_best + beta*sigma, improvement is negative. 
    # EI would be 0. 
    # For a "score" where higher is better, negative scores are bad.
    # We can clamp to 0 or leave as is. 
    # Leaving as is allows the function to strongly prefer points with high potential improvement.
    # If all improvements are negative (unlikely in early stages, possible in late stages if stuck),
    # the function will pick the least negative (closest to 0), which is correct.
    
    scores = improvement / safe_cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
