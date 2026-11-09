"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level cache for time management
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # 300s per evaluation, but we have 330s total wall clock for the whole run
# Actually, the prompt says "Each evaluation must finish within 300 s." and "evaluator KILLS the program after 330s wall-clock."
# The acquisition function is called repeatedly. We should ensure it's fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    We use a modified Expected Improvement (EI) that accounts for cost.
    The standard EI is: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    To make it cost-aware, we divide by the cost. However, a simple division by cost
    might be too aggressive if costs vary significantly. 
    A common approach is to use EI / cost, or perhaps a weighted combination.
    
    Given the problem:
    - Cost = 1 + mean(x_scaled). x_scaled is in [0,1]. So cost is in [1, 2].
    - We want to maximize the acquisition score.
    
    Let's stick with EI / cost for simplicity and effectiveness, as in the seed program,
    but let's check if there are better options.
    
    Alternative: Use a combination of EI and UCB (Upper Confidence Bound).
    UCB = mu + kappa * sigma. But we are minimizing, so we want low mu and high uncertainty.
    Actually, for minimization, UCB would be mu - kappa * sigma? No, that's for maximization.
    For minimization, we want to explore where the function might be low.
    So, a "lower confidence bound" LCB = mu - kappa * sigma would be good for exploration.
    But EI already balances exploitation and exploration.
    
    Let's try a different cost-aware metric.
    Maybe: EI / (cost^alpha) for some alpha < 1 to soften the cost penalty?
    Or: EI - lambda * cost?
    
    The seed program uses EI / cost. This is a standard approach.
    Let's consider the nature of the Levy 2D function. It has a global minimum and local minima.
    The GP model provides mu and sigma.
    
    One potential issue with EI / cost is that it heavily penalizes points with high cost.
    Since cost is 1 + mean(x_scaled), points with higher coordinates are more expensive.
    This might bias the search towards the lower-left corner (lower coordinates), 
    which might not contain the global minimum.
    
    To mitigate this, we could use a softer cost penalty.
    For example: EI / (1 + 0.5 * (cost - 1))
    Or: EI / sqrt(cost)
    
    Let's try EI / sqrt(cost). This reduces the impact of cost compared to EI / cost.
    
    Also, we should handle the case where sigma is very small to avoid division by zero.
    
    Let's also consider adding a small exploration term to encourage diversity.
    However, EI already includes exploration via the sigma term.
    
    Let's implement EI / sqrt(cost) and see if it performs better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment: divide by sqrt(cost) to soften the penalty
    # cost is in [1, 2], so sqrt(cost) is in [1, sqrt(2) ~ 1.414]
    # This is a less aggressive penalty than dividing by cost.
    scores = ei / np.sqrt(cost)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
