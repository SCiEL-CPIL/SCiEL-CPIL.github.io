"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Lower Confidence Bound (LCB) approach.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Mean of the GP at candidate points.
    sigma: np.ndarray of shape (N,) - Standard deviation of the GP at candidate points.
    f_best: float - The best (lowest) observed objective value so far.
    cost: np.ndarray of shape (N,) - Cost of evaluating each candidate.
    
    Returns:
    scores: np.ndarray of shape (N,) - Acquisition scores (higher is better).
    """
    # Add a small epsilon to avoid division by zero or negative costs if any
    # Cost is defined as 1 + mean(x_scaled), so it's always >= 1.
    safe_cost = np.maximum(cost, 1e-9)
    
    # LCB: mu - beta * sigma
    # We want to minimize f, so we want low mu and high sigma (exploration).
    # A common heuristic for beta is 1.0 or 2.0. 
    # Since we are minimizing, the "value" of a point is mu - beta*sigma.
    # We want to maximize the "improvement" relative to f_best, or just minimize the LCB.
    # However, the prompt asks for scores where "higher = evaluate next".
    # So we should return -LCB / cost, or (f_best - LCB) / cost, or similar.
    
    # Let's use a formulation similar to EI but based on LCB.
    # Standard LCB for minimization: a(x) = mu(x) - kappa * sigma(x)
    # We want to pick x with the lowest a(x).
    # So the score to maximize should be -a(x) / cost(x) = (kappa * sigma(x) - mu(x)) / cost(x)
    
    # However, a pure LCB might ignore f_best. 
    # A better cost-aware metric is often "Expected Improvement" or "Probability of Improvement".
    # But the directive asks for a DIFFERENT strategy.
    
    # Let's try: (f_best - mu) / sigma + kappa * sigma / f_best? No, that's messy.
    
    # Let's stick to a robust variant: 
    # Score = (f_best - mu) / (sigma + eps) + beta * sigma / (f_best - mu + eps) ? No.
    
    # Let's use a simpler, effective cost-aware LCB:
    # We want to minimize f. 
    # The "optimistic" value is mu - beta*sigma.
    # We want to maximize the potential improvement per unit cost.
    # Improvement potential = f_best - (mu - beta*sigma)
    # Score = (f_best - mu + beta*sigma) / cost
    
    # Choosing beta:
    # If beta is too high, we explore too much. If too low, we exploit too much.
    # A value around 1.0 to 2.0 is standard. Let's use 1.5.
    beta = 1.5
    
    # Calculate the optimistic value (lower is better for the function)
    optimistic_value = mu - beta * sigma
    
    # Calculate the potential improvement relative to f_best
    # If optimistic_value < f_best, there is potential improvement.
    # improvement = f_best - optimistic_value
    improvement = f_best - optimistic_value
    
    # We want to maximize this improvement per unit cost.
    # However, if improvement is negative, we might still want to explore if sigma is high?
    # Actually, if mu - beta*sigma > f_best, the LCB is worse than the current best.
    # EI handles this by being 0. LCB doesn't necessarily.
    # To ensure we only pick points that *might* be better, we can clip negative improvements to 0,
    # or just let the ratio handle it. If improvement is negative, score is negative.
    # Since we are maximizing, negative scores will be avoided if any positive exist.
    # If all are negative, it picks the least negative (closest to f_best).
    
    scores = improvement / safe_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
