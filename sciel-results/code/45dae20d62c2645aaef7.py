# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Combines:
    1. Expected Improvement (EI) - exploitation signal
    2. Cost-aware penalty: penalizes expensive evaluations, but scales with 
       how much improvement is expected (expensive is OK if EI is high)
    3. Uncertainty bonus: encourages exploration when model is uncertain,
       weighted by cost efficiency (cheap + uncertain = good exploration)
    
    The cost is 1 + mean(x_scaled) where x_scaled in [0,1], so cost in [1,2].
    """
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to get a relative measure of improvement potential
    # This helps balance the cost penalty across different scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max  # in [0, 1]
    
    # Cost penalty: 
    # - (cost - 1) is the extra cost beyond minimum, in [0, 1]
    # - Scale by sigma (uncertainty): uncertain expensive points are penalized more
    # - But reduce penalty when EI is high: if we expect big improvement, 
    #   the cost is justified
    # penalty = (cost - 1) * sigma * (1 - ei_norm * 0.7) * alpha
    # When ei_norm is high (good candidate), penalty is reduced by up to 70%
    # When ei_norm is low, full penalty applies
    
    alpha = 0.3  # base penalty strength
    cost_penalty = (cost - 1.0) * sigma * (1.0 - ei_norm * 0.7) * alpha
    
    # Uncertainty bonus:
    # Encourage exploring uncertain regions, especially cheap ones
    # bonus = sigma * (1 - (cost - 1)) * beta
    # Cheap + uncertain = high bonus, expensive + uncertain = low bonus
    beta = 0.15
    uncertainty_bonus = sigma * (2.0 - cost) * beta  # (2-cost) in [0,1], cheap=high
    
    # Combine: EI dominates, cost penalty discourages wasteful expensive evals,
    # uncertainty bonus encourages cheap exploration
    score = ei - cost_penalty + uncertainty_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
