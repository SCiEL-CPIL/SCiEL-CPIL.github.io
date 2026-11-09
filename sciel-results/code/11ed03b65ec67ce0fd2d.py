"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound
    # UCB encourages exploration of areas with high uncertainty (sigma)
    # and exploitation of areas with low predicted value (mu).
    # Dividing by cost penalizes expensive evaluations.
    
    # Beta controls the exploration-exploitation trade-off.
            # A value around 3-5 is typical for small budgets.
            beta = 3.5
            
            # Calculate UCB scores
            # We want to minimize f, so we maximize -mu + beta*sigma
            ucb = -mu + beta * sigma
            
            # Normalize by cost
            # cost is guaranteed to be > 0 (min 1.0)
            scores = ucb / cost
            
            # Ensure finite values
            scores = np.where(np.isfinite(scores), scores, -np.inf)
            
            return scores
        # EVOLVE-BLOCK-END
