"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-6)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # We want low mu - kappa * sigma, so score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # Use adaptive kappa based on exploration needs
    kappa = 1.5
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # Compute UCB (Upper Confidence Bound) variant - actually for minimization we want LCB
    # But let's also consider a PI (Probability of Improvement) based score
    pi_scores = norm.cdf(z)
    
    # Normalize each score to [0, 1] range for fair combination
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-10:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    ei_norm = normalize(ei_scores)
    lcb_norm = normalize(lcb_scores)
    pi_norm = normalize(pi_scores)
    
    # Hybrid strategy: weighted combination
    # EI provides exploitation, LCB provides exploration, PI is a simpler exploitation signal
    w_ei = 0.5
    w_lcb = 0.35
    w_pi = 0.15
    
    final_scores = w_ei * ei_norm + w_lcb * lcb_norm + w_pi * pi_norm
    
    # Check time
    if time.time() - start_time > deadline - start_time:
        return _sanitize(ei_scores)
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
