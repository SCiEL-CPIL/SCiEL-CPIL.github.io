"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# UCB acquisition function
# beta controls the exploration-exploitation trade-off.
# Higher beta -> more exploration (penalizes low sigma less, rewards high sigma more).
# For Branin 2D with 30 evals, a moderate to high beta is often beneficial to find all minima.
# The seed program uses EI (beta=0 equivalent in some contexts, or strictly exploitation).
# We use a fixed beta. A common heuristic is beta ~ 1.0 to 5.0.
# Given the small length scale (0.2) and standardized targets, sigma values might be small.
# Let's start with beta=2.0.

def acquisition(mu, sigma, f_best):
    # UCB: mu + beta * sigma
    # Note: We want to MINIMIZE the function, so we want high mu? 
    # Wait. 
    # The task says: "higher meaning evaluate next".
    # The evaluator minimizes branin 2d.
    # So the acquisition function should return high scores for points likely to have LOW f(x).
    # 
    # In minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is high when mu is low (close to f_best or lower) and sigma is moderate.
    #
    # UCB for minimization is typically: mu - beta * sigma
    # This is high when mu is low and sigma is high.
    # 
    # Let's verify the sign.
    # If we use UCB = mu - beta * sigma:
    # - If mu is very low (good), score is high.
    # - If sigma is very high (uncertain), score is high (exploration).
    # 
    # The current EI program:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If mu < f_best, f_best - mu > 0. EI > 0.
    # If mu > f_best, f_best - mu < 0. EI can be small or negative.
    #
    # So for minimization, we want to maximize the acquisition score.
    # UCB_min = mu - beta * sigma.
    
    beta = 2.0
    scores = mu - beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
