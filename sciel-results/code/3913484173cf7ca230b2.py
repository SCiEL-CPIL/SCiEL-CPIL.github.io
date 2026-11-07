"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Strategy: 
    1. Structural Change: Use a UCB (Upper Confidence Bound) based score, 
       which has shown superior performance over pure EI for this problem.
    2. Refinement: Make the exploration parameter (beta) adaptive based on 
       the variance of the predicted means (mu) across the candidate set.
       High variance in mu suggests a complex or unexplored landscape, 
       warranting more exploration. Low variance suggests a well-explored 
       region, warranting more exploitation.
    3. Additionally, include a small EI component to ensure that points 
       with high expected improvement are not overlooked if the UCB 
       signal is weak.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is strictly positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Calculate Standard EI (for reference/blend)
    z = (fb - mu) / sigma_safe
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Calculate Adaptive Beta for UCB
    # Base beta is 0.2 (from Inspiration 1, which scored well)
    # We increase beta if the spread of mu is large.
    # Normalize the variance to [0, 1] roughly.
    mu_var = np.var(mu)
    # Sigmoid scaling to map variance to [0, 1]
    # If mu_var is 0, factor is 0. If mu_var is large, factor is 1.
    var_factor = mu_var / (mu_var + 1.0)
    
    # Beta ranges from 0.2 to 0.7
    beta = 0.2 + 0.5 * var_factor

    # UCB Score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Ensure UCB scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)

    # Combine UCB and EI.
    # EI is in units of improvement (positive is good).
    # UCB is in units of objective value (negative is good, so we maximize -mu...).
    # To combine, we need to align scales.
    # A simple approach: Return the UCB score, but if EI is extremely high 
    # relative to the UCB score's range, we might want to blend.
    # However, mixing scales is risky.
    # Given Inspiration 1's success with pure UCB, we will primarily return UCB.
    # We will add a small bonus for EI to break ties or boost exploitation 
    # when UCB is flat.
    
    # Normalize EI to be comparable to UCB?
    # Max EI is roughly fb - min(mu).
    # Max UCB is roughly -min(mu) + beta*max(sigma).
    
    # Let's try a weighted sum where weights are fixed but small for EI.
    # score = ucb_score + 0.1 * ei
    
    final_score = ucb_score + 0.1 * ei
    
    # Final safety check
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)

    return final_score
# EVOLVE-BLOCK-END
