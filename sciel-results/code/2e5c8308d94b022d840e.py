import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with an adaptive exploration term based on posterior uncertainty.
    
    Args:
        mu: numpy array of shape (N,), posterior mean.
        sigma: numpy array of shape (N,), posterior standard deviation.
        f_best: float, best observed value so far.
        cost: numpy array of shape (N,), cost of evaluating each point.
        
    Returns:
        numpy array of shape (N,), acquisition scores (higher is better).
    """
    # Ensure arrays are 1D and of correct type
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Due to numerical precision, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI score
    score_ei = ei / cost_safe
    
    # Exploration term based on posterior uncertainty
    # Normalize sigma to [0, 1] to make it comparable across iterations
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive weight for exploration:
    # When f_best is high (early in optimization), we want more exploration.
    # When f_best is low (late in optimization), we want less exploration.
    # w = 1 / (1 + f_best / scale)
    # If f_best is large, w is small (less exploration).
    # If f_best is small, w is close to 1 (more exploration).
    # This is the opposite of what we want. We want MORE exploration early.
    # So let's flip it: w = f_best / (1 + f_best / scale)
    # If f_best is large, w is close to scale (more exploration).
    # If f_best is small, w is small (less exploration).
    
    # Let's try a different approach.
    # The scale of EI is roughly f_best. The scale of sigma is also related to f_best.
    # Let's use a fixed small weight for exploration, but scale it by the normalized sigma.
    # And we can make the weight depend on how "good" f_best is.
    # If f_best is very good (close to 0), we can afford to be more exploitative.
    # If f_best is bad, we should explore more.
    
    # Let's use a weight that is high when f_best is high.
    # w = 1.0 / (1.0 + np.exp(-f_best / 5.0))
    # This is a sigmoid.
    # f_best = 0 -> 0.5
    # f_best = 10 -> 0.88
    # f_best = 20 -> 0.98
    
    # Actually, the inspiration used: w = 1.0 / (1.0 + f_best / 5.0)
    # f_best = 0 -> 1.0
    # f_best = 5 -> 0.5
    # f_best = 20 -> 0.2
    # This means MORE exploration when f_best is LOW. That seems wrong.
    # We want to explore when we are far from the optimum (f_best is HIGH).
    
    # Let's use: w = f_best / (f_best + 5.0)
    # f_best = 0 -> 0.0
    # f_best = 5 -> 0.5
    # f_best = 20 -> 0.8
    # This means MORE exploration when f_best is HIGH. This makes more sense.
    
    w = f_best / (f_best + 5.0)
    
    # Scale the exploration term.
    # The exploration term should be small compared to EI when EI is large.
    # Let's use a small base weight, say 0.1.
    # score_explore = 0.1 * w * sigma_norm
    
    # But we also need to divide by cost to make it cost-aware.
    # score_explore = (0.1 * w * sigma_norm) / cost_safe
    
    # Let's combine them.
    score = score_ei + (0.1 * w * sigma_norm) / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
