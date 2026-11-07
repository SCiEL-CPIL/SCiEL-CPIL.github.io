import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with normalized exploration and adaptive cost penalty.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Scalar, lowest observed function value
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to ensure numerical stability
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1] to balance with exploration term
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # Calculate Normalized Exploration Term
    # Scale-invariant uncertainty measure
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        exploration = sigma / sigma_max
    else:
        exploration = np.zeros_like(sigma)
        
    # Adaptive beta: balance exploitation (EI) and exploration (uncertainty)
    # Early on (high uncertainty spread), favor exploration more
    # Later (low uncertainty spread), favor exploitation more
    sigma_spread = np.std(sigma)
    if sigma_max > 1e-8:
        spread_ratio = sigma_spread / sigma_max
    else:
        spread_ratio = 0.0
    
    # Map spread_ratio [0, 1] to beta [0.5, 2.0]
    # High spread -> high beta (more exploration)
    # Low spread -> low beta (more exploitation)
    beta = 0.5 + 1.5 * np.clip(spread_ratio, 0, 1)
    
    # Combine EI and exploration
    base_score = ei_norm + beta * exploration
    
    # Apply adaptive cost penalty
    # Cost is in [1, 2]. 
    # Use a penalty that's stronger for expensive points but not too aggressive
    # cost^-0.7 maps [1, 2] to [1.0, ~0.616]
    # This is stronger than cost^-0.5 but less aggressive than 1/cost
    cost_penalty = np.power(cost, -0.7)
    
    scores = base_score * cost_penalty
    
    # Deadline guard: return safe fallback if computation is unexpectedly slow
    # The computation is O(N) and very fast, but we adhere to the directive.
    if time.time() - t_start > 2.0:
        return exploration * cost_penalty
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
