"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0  # 2 seconds safety margin out of 300s total budget per call? 
    # Actually, the constraint is 300s for the *evaluation* of the best candidate? 
    # "Each evaluation must finish within 300 s." 
    # The acquisition function is called *before* evaluation. 
    # The evaluator scores 2000 candidates. 
    # The total time for the *loop* is likely the constraint, but the prompt says "Each evaluation must finish within 300 s".
    # Usually, the acquisition function is fast. 
    # However, the directive says: "the evaluator KILLS the program after 330s wall-clock." 
    # So we have ~330s total. 
    # We should keep acquisition very fast.
    
    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Structural Change: Cost-aware UCB (Upper Confidence Bound) for minimization
    # We maximize the potential improvement, which is f_best - (mu - beta * sigma)
    # This encourages exploration (high sigma) and exploitation (low mu).
    # We divide by cost to account for the cost-awareness.
    
    # Beta controls the exploration/exploitation trade-off.
    # In 8D, we need significant exploration. 
    # Objective range is roughly [-0.8, 0.8]. Sigma is typically small.
    # Let's use a beta that scales with the uncertainty.
    # A fixed beta of 1.0 might be too aggressive if sigma is small, but if sigma is large, it's good.
    # Let's try beta = 1.0.
    beta = 1.0
    
    ucb_score = (f_best - mu) + beta * sigma
    scores = ucb_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
