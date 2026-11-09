import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d).
    
    Uses a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Cost-normalized LCB for exploration
    - Adaptive weighting based on uncertainty
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Deadline check
    def time_left():
        return 330.0 - (time.time() - t0) - 5.0
    
    if time_left() <= 0:
        # Fallback: simple LCB
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) 
    #                         + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is CDF and phi is PDF of standard normal
    
    z = (f_best - mu) / sigma
    
    # Use stable computation
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI for minimization
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0)  # EI is non-negative
    
    # Cost-normalized EI
    ei_cost = ei / cost
    
    # Compute LCB (Lower Confidence Bound)
    # L(x) = mu(x) - kappa * sigma(x)
    # We want to minimize L, so score = -L / cost = (-mu + kappa*sigma) / cost
    
    # Adaptive kappa: larger when uncertainty is high
    # Use a base kappa that adapts based on the scale of sigma
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Base kappa with slight adaptation
    kappa_base = 1.5
    # Increase kappa when there's more uncertainty relative to the function scale
    uncertainty_ratio = sigma_max / (abs(f_best) + 1.0)
    kappa = kappa_base + 0.5 * np.clip(uncertainty_ratio, 0, 1.0)
    
    lcb_score = (-mu + kappa * sigma) / cost
    
    # Hybrid: blend EI and LCB
    # Weight EI more when we have a good best value (exploitation)
    # Weight LCB more when uncertainty is high (exploration)
    
    # Measure how "confident" we are: if sigma is small relative to improvement potential
    # Use a simple blend
    w_ei = 0.6
    w_lcb = 0.4
    
    scores = w_ei * ei_cost + w_lcb * lcb_score
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Normalize scores to have zero mean and unit variance for stability
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    if score_std > 1e-10:
        scores = (scores - score_mean) / score_std
    
    return scores
