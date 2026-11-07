"""Cost-aware acquisition for 2D Ackley BO (Adaptive Cost-Weighted LCB)."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for each candidate.
    Higher is better (means "evaluate next").

    We use an Adaptive Lower Confidence Bound (LCB) normalized by cost.
    Score = (beta * sigma - mu) / cost
    
    The beta parameter is adaptive: it starts high (exploration) when f_best is poor 
    and decreases (exploitation) as f_best approaches the minimum.
    """
    n = mu.shape[0]
    eps = 1e-9

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=eps, posinf=1e6, neginf=eps)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Cost is always >= 1 in this problem, but ensure safety
    cost = np.maximum(cost, eps)
    sigma = np.maximum(sigma, eps)

    # Adaptive beta:
    # When f_best is high (poor), exp(-f_best) is small -> beta is low?
    # Wait. If f_best is 10, we want MORE exploration.
    # If f_best is 0, we want LESS exploration.
    # So beta should be HIGH when f_best is HIGH?
    # No, usually we explore when we know little. 
    # If f_best is 10, the function values are generally high. We need to find the basin.
    # If f_best is 0.1, we are close. We need to refine.
    # So beta should decrease as f_best decreases.
    
    # Let's use: beta = beta_min + (beta_max - beta_min) * exp(-f_best)
    # If f_best = 0, exp(0)=1 -> beta = beta_max
    # If f_best = 10, exp(-10)~0 -> beta = beta_min
    # This means we explore MORE when f_best is GOOD? That seems backwards.
    
    # Let's rethink.
    # Standard UCB: x = argmax (mu + beta * sigma).
    # If we are far from the optimum, sigma is large everywhere.
    # As we get closer, sigma decreases in the explored area.
    # The "uncertainty" naturally drops.
    # But if we want to force exploration early, we need a high beta.
    # How do we know if we are "early"? f_best is high.
    # So if f_best is high, we should have a HIGH beta?
    # If f_best is 10, and we use beta=10, we push hard for high sigma.
    # If f_best is 0.1, and we use beta=0.1, we push for low mu.
    
    # So: beta = beta_min + (beta_max - beta_min) * (f_best / f_max_estimate)
    # Or simply: beta = 1.0 + 2.0 * f_best
    # If f_best = 0, beta = 1.0.
    # If f_best = 5, beta = 11.0.
    # This encourages exploration when the current best is bad.
    
    # Let's try this linear scaling.
    # Ackley max is ~30.
    # beta = 1.0 + 0.5 * f_best
    # If f_best = 0, beta = 1.0.
    # If f_best = 10, beta = 6.0.
    # If f_best = 30, beta = 16.0.
    
    beta = 1.0 + 0.5 * f_best
    
    # Calculate LCB
    lcb = -mu + beta * sigma
    
    # Cost-normalized LCB
    scores = lcb / cost

    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
