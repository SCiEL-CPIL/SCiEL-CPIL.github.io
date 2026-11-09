"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) acquisition.
    # Higher scores mean more promising (lower expected value with uncertainty penalty).
    # We want to minimize, so LCB = mu - beta * sigma.
    # Returning LCB directly means higher LCB scores are preferred.
    # But wait: the task says "higher meaning evaluate next".
    # For minimization, we want to find low values.
    # LCB = mu - beta * sigma gives a lower bound on the objective.
    # We want to pick the candidate with the lowest possible objective,
    # so we should maximize the "optimistic" lower bound.
    # Actually, for minimization:
    #   EI picks points where we expect improvement over f_best.
    #   LCB for minimization: we want to minimize mu - beta*sigma? No.
    #   Let's think: if we believe the true value is around mu, and sigma is uncertainty,
    #   the lower bound on the true value is mu - beta*sigma (pessimistic lower bound).
    #   We want to find the minimum, so we should pick the point with the lowest lower bound?
    #   No, that's wrong. If the lower bound is low, it means the true value could be very low.
    #   So we want to MINIMIZE mu - beta*sigma? That doesn't make sense either.
    
    # Let me reconsider. For minimization:
    #   The expected value is mu. The uncertainty is sigma.
    #   A lower confidence bound on the objective value is mu - beta*sigma.
    #   If this bound is low, it suggests the true value might be low (good for minimization).
    #   So we want to pick the candidate with the lowest mu - beta*sigma.
    #   But the task says "higher meaning evaluate next".
    #   So we should return something where higher is better.
    #   We can return -(mu - beta*sigma) = -mu + beta*sigma.
    #   Or equivalently, return beta*sigma - mu.
    
    # Let's use a standard LCB approach for minimization:
    #   score = -mu + beta * sigma
    #   Higher score means we prefer this point.
    
    # Choose beta. A common choice is beta = 2 or so, but let's try a fixed value.
    # Since mu and sigma are in objective units, and f_best is the current best,
    # we can scale beta relative to the problem scale.
    # The Michalewicz function has values roughly in [-1.8, 1.0].
    # Let's use beta = 2.0 as a starting point.
    
    beta = 2.0
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
