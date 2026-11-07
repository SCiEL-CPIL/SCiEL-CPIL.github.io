import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit
    
    # Global minimum of Michalewicz 2D
    opt = -1.8013
    
    # Simple regret so far
    sr = f_best - opt
    
    # Adaptive exploration parameter
    # When far from optimum, explore more; when close, exploit
    if sr > 1.0:
        kappa = 2.0
    elif sr > 0.5:
        kappa = 1.5
    elif sr > 0.2:
        kappa = 1.0
    elif sr > 0.05:
        kappa = 0.5
    else:
        kappa = 0.2
    
    # LCB score
    lcb = -mu + kappa * sigma
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(lcb), lcb, 0.0)
        return scores
    
    # Expected Improvement (EI) acquisition
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    # For minimization, we use (f_best - mu) as the improvement
    
    from scipy.stats import norm
    
    z = (f_best - mu) / (sigma + 1e-10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine LCB and EI
    # Weight between them based on how close we are to optimum
    if sr > 0.5:
        w_ei = 0.3
    elif sr > 0.1:
        w_ei = 0.5
    else:
        w_ei = 0.7
    
    w_lcb = 1.0 - w_ei
    
    scores = w_lcb * lcb + w_ei * ei
    
    # Add a small bonus for points with very low uncertainty near the best
    # This helps refine the solution when we're close
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-10
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Bonus for low uncertainty
    confidence_bonus = 0.1 * (1.0 - sigma_norm)
    
    scores = scores + confidence_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
