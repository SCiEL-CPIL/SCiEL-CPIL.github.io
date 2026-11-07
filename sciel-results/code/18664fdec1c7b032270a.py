"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization.
    # LCB = mu - beta * sigma
    # We want to minimize, so lower LCB is better.
    # To make it "higher is better" for the acquisition score, we can just return LCB / cost?
    # Wait, the task says "Return 2000 finite scores (higher = evaluate next)".
    # If we use LCB for minimization, a low LCB is good.
    # So we should return -LCB / cost? Or (f_best - LCB) / cost?
    # Actually, usually acquisition functions are defined such that higher is better.
    # EI is higher when improvement is expected.
    # LCB for minimization: we want the lowest value.
    # So score = - (mu - beta * sigma) / cost?
    # Or score = (beta * sigma - mu) / cost?
    
    # Let's check the seed program:
    # return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    # This is EI / cost. EI is positive when mu < f_best. Higher EI is better.
    
    # So for LCB, we need a score that is high when the lower bound is low.
    # Score = - (mu - beta * sigma) / cost
    # Or Score = (beta * sigma - mu) / cost
    
    beta = 5.0
    # Calculate LCB (lower bound on the minimum)
    lcb = mu - beta * sigma
    
    # We want to maximize the score, so we want to minimize lcb.
    # Thus, score = -lcb / cost
    scores = -lcb / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
