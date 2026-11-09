import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a Knowledge Gradient (KG) inspired acquisition that combines:
    1. Expected Improvement (EI): P(f(x) < f_best) * (f_best - E[f(x)|f(x)<f_best])
    2. Exploration bonus: rewards high-uncertainty regions to discover new minima
    
    This global view better identifies points that maximize the probability
    of locating the true minimum within the remaining budget.
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-scores: how many std devs below f_best is each candidate
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50.0, 50.0)
    
    # Probability that candidate improves upon f_best
    prob_improve = norm.cdf(z)
    
    # Expected value conditional on improvement: mu - sigma * phi(z) / Phi(z)
    # For numerical stability, handle cases where prob_improve is very small
    phi_z = norm.pdf(z)
    # Avoid division by zero
    safe_prob = np.where(prob_improve > 1e-15, prob_improve, 1e-15)
    cond_mean = mu - sigma * phi_z / safe_prob
    
    # Expected Improvement: prob_improve * (f_best - cond_mean)
    # This equals prob_improve * (f_best - mu + sigma * phi(z)/Phi(z))
    # Which simplifies to: prob_improve * (f_best - mu) + sigma * phi(z)
    ei = prob_improve * (f_best - mu) + sigma * phi_z
    
    # Exploration bonus: reward points with high uncertainty
    # Scale by how far f_best is from 0 (the known minimum)
    fb = np.clip(f_best, 0.0, 1.0)
    
    # Adaptive exploration weight:
    # Early on (high f_best), explore more. Late on, exploit more.
    explore_weight = 0.5 + 1.5 * fb
    
    # Normalize sigma for the exploration bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # KG-inspired score: EI + exploration bonus
    score = ei + explore_weight * sigma_norm * fb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
