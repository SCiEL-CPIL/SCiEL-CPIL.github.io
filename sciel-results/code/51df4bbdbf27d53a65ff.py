# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _elapsed():
    return time.time() - _ensure_start_time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a 
    Knowledge Gradient (KG) approximation based on the probability of 
    being the global minimum among the candidates.
    
    The EI term exploits regions likely to beat f_best.
    The KG term (via probability of being global min) explores regions 
    that are likely to contain the true global minimum, helping to 
    avoid local optima in multi-modal functions.
    """
    # Deadline guard
    if _elapsed() > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Probability of Being the Global Minimum (KG Approximation)
    # Assuming independence between candidates for this approximation.
    # P(i is min) = prod_{j != i} P(f_i < f_j)
    # P(f_i < f_j) = Phi((mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2))
    
    n = len(mu)
    
    # Create matrices for differences and combined stds
    # mu_diff[i, j] = mu[i] - mu[j]
    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    
    # sigma_comb[i, j] = sqrt(sigma[i]^2 + sigma[j]^2)
    sigma_comb = np.sqrt(sigma_safe[:, np.newaxis]**2 + sigma_safe[np.newaxis, :]**2)
    
    # Avoid division by zero (though sigma_safe > 0)
    sigma_comb = np.maximum(sigma_comb, 1e-9)
    
    # Compute probabilities P(f_i < f_j)
    # z_ij = (mu_i - mu_j) / sigma_comb_ij
    z_ij = mu_diff / sigma_comb
    
    # P(f_i < f_j) = Phi(z_ij)
    p_ij = norm.cdf(z_ij)
    
    # For j=i, P(f_i < f_i) is 0.5, but we want the probability of being the minimum,
    # so we exclude j=i from the product. We set diagonal to 1.
    np.fill_diagonal(p_ij, 1.0)
    
    # Compute log probability of being the minimum
    # log(P(i is min)) = sum_{j != i} log(P(f_i < f_j))
    log_p_min = np.sum(np.log(p_ij + 1e-300), axis=1)
    
    # Convert back to probability
    p_min = np.exp(log_p_min)
    
    # Ensure p_min is finite and valid
    p_min = np.where(np.isfinite(p_min), p_min, 0.0)
    
    # 3. Combine EI and KG
    # The KG term should be scaled to be comparable to EI.
    # A reasonable scale is the potential improvement, e.g., (f_best - min(mu))
    potential_improvement = f_best - np.min(mu)
    if potential_improvement < 0:
        potential_improvement = 0.0
        
    # Weight for the KG term
    # We use a small weight to blend with EI
    kg_weight = 0.1
    
    kg_bonus = kg_weight * potential_improvement * p_min
    
    scores = ei + kg_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
