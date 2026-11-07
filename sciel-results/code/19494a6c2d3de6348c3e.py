import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Calculate Expected Improvement (EI) for minimization.
    2. Normalize EI by the cost of evaluation to prioritize cheap improvements.
    3. Add a small exploration bonus (UCB-like) to encourage visiting uncertain regions,
       scaled by the square root of the number of evaluations (approximated via cost or just fixed).
       Since we don't have 'n' directly, we rely on the fact that early on sigma is high.
       A fixed small beta for exploration ensures we don't get stuck in local minima 
       that EI might ignore if the GP is slightly miscalibrated.
    4. Ensure all scores are finite.
    """
    
    # Safety guard: Check time to avoid timeout
    # The prompt says "Each evaluation must finish within 300 s".
    # This function is called many times. It should be fast.
    # NumPy operations on 2000 elements are microseconds.
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace NaNs and Infs
    # If mu is NaN, it's likely an error in GP prediction. Assume it's f_best (no improvement).
    # If sigma is NaN or Inf, assume it's 0 (no uncertainty) or large. 
    # Let's be conservative: NaN mu -> f_best, NaN sigma -> 0.
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=0.0)
    sigma_safe = np.maximum(sigma_safe, 1e-6) # Avoid division by zero in Z calculation
    
    cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    cost_safe = np.maximum(cost_safe, 1e-6) # Avoid division by zero
    
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is CDF of standard normal, phi is PDF of standard normal.
    
    # Handle case where sigma is very small to avoid numerical issues in Z
    # If sigma is tiny, Z is huge if mu < f_best, or -huge if mu > f_best.
    # Phi(huge) = 1, phi(huge) = 0 -> EI = f_best - mu
    # Phi(-huge) = 0, phi(-huge) = 0 -> EI = 0
    
    # We compute Z
    Z = (f_best - mu_safe) / sigma_safe
    
    # To avoid overflow/underflow in exp for phi, we can use log-pdf or clip Z.
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    # If |z| > 40, phi(z) is effectively 0.
    # We can clip Z for the phi calculation, but use raw Z for Phi.
    # Actually, scipy.stats.norm.cdf and pdf are robust.
    
    # Clip Z to [-50, 50] to be safe for pdf, though cdf is fine.
    Z_clip = np.clip(Z, -50, 50)
    
    phi_Z = norm.pdf(Z_clip)
    Phi_Z = norm.cdf(Z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If mu > f_best, (f_best - mu) is negative.
    # Phi(Z) will be small (since Z is negative).
    # The term (f_best - mu)*Phi(Z) is negative * small = small negative?
    # Wait. The formula EI = (f_best - mu)Phi(Z) + sigma phi(Z) is for minimization?
    # Let's check.
    # f(x) ~ N(mu, sigma^2). We want f(x) < f_best.
    # Let Y = f_best - f(x). We want Y > 0.
    # Y ~ N(f_best - mu, sigma^2).
    # E[max(Y, 0)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma).
    # Yes, this is correct.
    
    EI = (f_best - mu_safe) * Phi_Z + sigma_safe * phi_Z
    
    # EI should be non-negative. Due to floating point errors, it might be slightly negative.
    EI = np.maximum(EI, 0.0)
    
    # Cost-awareness: Divide by cost
    # Higher score = higher EI per cost
    score_ei = EI / cost_safe
    
    # Exploration bonus
    # We add a small bonus for high uncertainty to ensure we don't miss regions 
    # where the GP is uncertain but current mu looks bad.
    # Standard UCB: mu - beta * sigma.
    # Here we are maximizing EI. Adding uncertainty directly helps exploration.
    # Let's add a small multiple of sigma.
    # How much? 0.1 * sigma seems reasonable. It shouldn't dominate EI but should break ties.
    beta_explore = 0.1
    exploration_bonus = beta_explore * sigma_safe / cost_safe
    
    scores = score_ei + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
