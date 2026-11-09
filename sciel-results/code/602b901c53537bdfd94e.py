"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of computing a deterministic score like Expected Improvement (EI)
    or Upper Confidence Bound (UCB), we sample a function value from the
    posterior Gaussian Process for each candidate point. We then return the
    negative of these samples as the score. The evaluator selects the candidate
    with the highest score, which corresponds to the lowest sampled function
    value.
    
    This approach is structurally different from optimization-based acquisition
    functions. It naturally balances exploration and exploitation:
    - Exploitation: Points with low mu are likely to have low samples.
    - Exploration: Points with high sigma have a wider distribution of samples,
      increasing the chance of drawing a very low value.
    
    Args:
        mu (np.ndarray): Mean of the GP posterior, shape (N,).
        sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,).
        f_best (float): The best (lowest) observed function value. Not used in
                        standard Thompson Sampling, but kept for interface
                        compatibility.
    
    Returns:
        np.ndarray: Scores for each candidate, shape (N,). Higher is better.
    """
    # Ensure sigma is strictly positive to avoid numerical issues in sampling
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw samples from the posterior N(mu, sigma^2)
    # Using standard normal random numbers for efficiency
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # We are minimizing the function.
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled values.
    # A low function value (good) becomes a high score (good).
    return -samples
# EVOLVE-BLOCK-END
