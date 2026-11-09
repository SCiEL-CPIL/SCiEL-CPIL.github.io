import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Being Optimal (PBO).
    Treats the set of candidates as a virtual population and scores each
    candidate by the log-probability of it being the minimum value in the set,
    assuming independence of the posterior samples for simplicity and speed.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback: return -mu (exploitation)
        return -mu
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute pairwise standard deviations: sigma_ij = sqrt(sigma_i^2 + sigma_j^2)
    # Using broadcasting
    sigma_sq_i = sigma_safe[:, np.newaxis]**2
    sigma_sq_j = sigma_safe[np.newaxis, :]**2
    sigma_pairwise = np.sqrt(sigma_sq_i + sigma_sq_j)
    
    # Compute pairwise mean differences: mu_j - mu_i
    # Shape (N, N) where row i, col j is mu_j - mu_i
    mu_diff = mu[np.newaxis, :] - mu[:, np.newaxis]
    
    # Z-score: (mu_j - mu_i) / sigma_ij
    # We want P(mu_i + eps_i < mu_j + eps_j)
    # This is P(eps_i - eps_j < mu_j - mu_i)
    # eps_i - eps_j ~ N(0, sigma_i^2 + sigma_j^2)
    # So Z = (mu_j - mu_i) / sigma_pairwise
    # P = Phi(Z)
    
    # Avoid division by zero (sigma_pairwise is >= 1e-9)
    z = mu_diff / sigma_pairwise
    
    # Clip z to avoid numerical issues in CDF
    z = np.clip(z, -10, 10)
    
    # Probability that i is better than j
    p_ij = norm.cdf(z)
    
    # We want the probability that i is better than ALL j != i
    # Product over j != i of p_ij
    # In log space: Sum over j != i of log(p_ij)
    
    # Set diagonal to 1 (log 1 = 0) because i is not better than itself in this context
    # (We are looking for the unique minimum)
    diag_idx = np.diag_indices(n)
    p_ij[diag_idx] = 1.0
    
    # Calculate log scores
    # log(p) can be -inf if p is 0.
    # np.log(0) is -inf.
    # Summing -inf gives -inf.
    # exp(-inf) is 0.
    # We want to return the log scores directly to avoid underflow in the final exp.
    # The evaluator picks the max. Max log-score is best.
    
    with np.errstate(divide='ignore'):
        log_p = np.log(p_ij)
        
    log_scores = np.sum(log_p, axis=1)
    
    # Replace -inf with a very small number to ensure finite output?
    # The task says "Return 2000 finite scores".
    # -inf is not finite.
    # So we must clip or replace.
    # A score of -1000 is effectively 0 probability.
    log_scores = np.where(np.isfinite(log_scores), log_scores, -1000.0)
    
    return log_scores
