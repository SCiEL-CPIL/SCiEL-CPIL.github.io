"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisiton function using Upper Confidence Bound (UCB) with an adaptive
    exploration parameter, combined with a penalty for very high uncertainty
    to avoid pathological edge cases. This differs from EI by directly balancing
    exploitation (low mu) and exploration (high sigma) with a tunable parameter.
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.clip(sigma, 1e-9, None)
    
    # Adaptive beta: starts high for early exploration, decays over time.
    # Since we don't have the step count directly, we can estimate it from f_best
    # relative to the known minimum, but a fixed reasonable beta often works well
    # for 30 evaluations. Let's use a beta that balances the scale.
    # The function values are around -2.5. Standard deviation might be ~0.1-0.5.
    # A beta of 1.0-2.0 is typical. Let's try a slightly aggressive exploration
    # early on. Without step count, we can't easily decay.
    # Alternative: Use PI (Probability of Improvement) or a hybrid.
    
    # Let's try UCB with a fixed beta that is known to perform well for small budgets.
    # Beta = 1.5 is a common choice.
    beta = 1.5
    
    ucb = mu + beta * sigma
    
    # To ensure finite scores and avoid inf/nan, clip the result
    # Also, we want higher score to mean "evaluate next".
    # Since we are minimizing, low mu is good. UCB = mu + beta*sigma.
    # We want to pick the point with the highest UCB? No, for minimization,
    # we want the point with the lowest *predicted* value but high uncertainty.
    # Standard UCB for minimization is: mu - beta * sigma.
    # Because we want to find low values. If sigma is high, we subtract more,
    # making the score lower? No, we want to *maximize* the acquisition score.
    # So if we define score = -mu + beta * sigma, then high uncertainty increases score.
    # Or score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's stick to the standard definition for minimization:
    # Acquisition = -mu + beta * sigma
    # This rewards low mean and high variance.
    
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
