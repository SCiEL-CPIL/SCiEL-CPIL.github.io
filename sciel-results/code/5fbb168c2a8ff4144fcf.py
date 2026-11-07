"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Global Best-of-N Acquisition Function.
    
    This strategy approximates Thompson Sampling by computing the probability
    that each candidate is the global minimum among the 2000 candidates,
    assuming independent posterior samples for each candidate (ignoring
    spatial correlations for speed and structural simplicity).
    
    Score_i = log( P(y_i < y_j for all j != i) )
    """
    n = len(mu)
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Add small epsilon to sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Compute pairwise Z-scores: Z_ij = (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2)
    # Shape: (n, n)
    mu_diff = mu[:, None] - mu[None, :]
    sigma_diff = np.sqrt(sigma_safe[:, None]**2 + sigma_safe[None, :]**2)
    
    # Avoid division by zero if both sigmas are extremely small (though eps helps)
    z = np.divide(mu_diff, sigma_diff, out=np.zeros_like(mu_diff), where=(sigma_diff > 1e-12))
    
    # Probability that i is better (lower) than j: P(y_i < y_j) = Phi(z_ij)
    # Note: If z is large positive, prob is ~1. If large negative, ~0.
    prob_win = norm.cdf(z)
    
    # Log probabilities to avoid underflow when multiplying
    # P(y_i < y_j) is in [0, 1]. Log is <= 0.
    log_prob_win = np.log(prob_win + 1e-15)
    
    # Exclude self-comparison: P(y_i < y_i) is 0.5, but we want to ignore self.
    # Set diagonal to 0 in log-space (which corresponds to 1.0 in prob-space)
    diag_idx = np.diag_indices(n)
    log_prob_win[diag_idx] = 0.0
    
    # Sum log-probabilities across all other candidates
    # This gives log( P(y_i is the minimum of the batch) )
    log_scores = np.sum(log_prob_win, axis=1)
    
    # The evaluator wants "higher meaning evaluate next".
    # log_scores are negative. Higher (closer to 0) is better.
    return log_scores
# EVOLVE-BLOCK-END
