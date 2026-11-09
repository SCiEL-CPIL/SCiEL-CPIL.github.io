"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to enforce the 330s wall-clock limit with a safety margin
_start_time = time.time()
TIME_LIMIT = 320.0 # Leave 10s buffer for cleanup/return

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient (KG) acquisition function.
    KG explicitly models the expected reduction in the minimum observed value,
    providing a more principled measure of information gain than EI.
    Score = KG(x) / cost(x)
    """
    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    f_best_val = float(f_best)
    
    # For Knowledge Gradient, we want to maximize the expected improvement
    # in the best observed value. The KG for a single candidate x is:
    # KG(x) = E[max(f_best, y(x))] - min(f_best, E[y(x)])
    # 
    # A practical approximation uses a two-point quadrature or the following
    # closed-form-like expression for the expected minimum:
    # 
    # The expected value of min(f_best, y) where y ~ N(mu, sigma^2) is:
    # E[min(f_best, y)] = f_best - (f_best - mu) * Phi(z) - sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # 
    # The Knowledge Gain is:
    # KG(x) = f_best - E[min(f_best, y)]
    #       = (f_best - mu) * Phi(z) + sigma * phi(z)
    # 
    # This is actually identical to Expected Improvement! However, KG can be
    # extended to account for multiple future evaluations or uncertainty in
    # which point will be best. Here we use an enhanced KG that includes
    # an exploration bonus based on the probability of finding a new minimum.
    
    # Calculate z-score
    z = (f_base_val - mu) / sigma_safe if False else (f_best_val - mu) / sigma_safe
    
    # Base KG (same as EI for single evaluation)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    kg_base = (f_best_val - mu) * cdf_val + sigma_safe * pdf_val
    kg_base = np.maximum(kg_base, 0.0)
    
    # Enhanced KG: add a term that rewards high uncertainty when the
    # predicted mean is close to f_best. This encourages exploration
    # in regions where the GP is uncertain but could potentially find
    # a better minimum.
    # The probability that y(x) < f_best is Phi(z). We weight the
    # uncertainty by this probability to focus exploration where
    # it's most likely to yield improvement.
    prob_improve = cdf_val
    uncertainty_bonus = 0.3 * sigma_safe * prob_improve
    
    # Combine base KG with uncertainty bonus
    scores_raw = kg_base + uncertainty_bonus
    
    # Cost-aware score: divide by evaluation cost to prioritize
    # high-value, low-cost regions
    scores = scores_raw / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
