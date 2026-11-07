"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_LIMIT = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Conditional Exploration.
    
    Strategy:
    - Primary signal: Expected Improvement (EI) divided by cost.
    - Secondary signal: Variance-based exploration bonus, but only triggered
      when EI is low (indicating the GP is confident in the current best region).
    - This creates a "greedy" search that explores only when stuck.
    - Includes a deadline guard.
    """
    # Deadline Guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_LIMIT:
        # Return a simple, fast score to ensure we don't timeout
        # Fallback to pure EI/cost or just random if needed
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10.0, 10.0)
        ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei_term = np.maximum(ei_term, 0.0)
        scores = ei_term / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Base score: EI per unit cost
    base_score = ei_term / cost_safe
    
    # 2. Compute Exploration Bonus
    # We want to explore only when EI is low.
    # Let's define a threshold for "low EI".
    # If EI is below a certain quantile of the EI distribution, we add variance.
    # A simple heuristic: if EI is less than 0.1 * (f_best - min_mu), explore.
    # Or simpler: Use a soft switch.
    
    # Exploration term: sigma / cost
    # We weight it by how "confident" we are.
    # If z is large (mu << f_best), EI is high, we don't need exploration.
    # If z is small/negative (mu >= f_best), EI is low, we need exploration.
    
    # The switch: weight = exp(-z^2 / 2) is not quite right.
    # Let's use the fact that EI is already low when z is small.
    # We can add a term proportional to sigma * (1 - EI_normalized).
    
    # Alternative: Use a "Probability of Improvement" (PI) based switch.
    # PI = norm.cdf(z). If PI is low, we are likely not improving.
    # Let's use a dynamic beta for UCB.
    # beta = 1.0 if EI is high, 3.0 if EI is low.
    
    # Calculate EI quantile
    # To keep it vectorized and fast, we use a simple threshold on z.
    # If z < 0.5, we consider it "low improvement" and boost exploration.
    
    # Exploration weight: higher when z is low (mu close to or above f_best)
    # weight = 1 / (1 + z^2)  -- smooth decay
    # But we want it to be 0 when z is very high (exploit) and 1 when z is low (explore).
    # Actually, EI is 0 when z <= 0. So if z <= 0, EI=0.
    # We should explore when EI is 0.
    
    # Let's use a hard switch with smoothing:
    # explore_weight = 1.0 where z < 0.5, fading to 0.0 where z > 2.0
    
    z_for_weight = np.clip(z, -5.0, 5.0)
    # Smooth step function: 1 at z=-5, 0 at z=5? No, we want 1 at low z, 0 at high z.
    # Let's use: weight = 1 / (1 + np.exp(z_for_weight))
    # This is 0.5 at z=0, close to 1 at z=-5, close to 0 at z=5.
    explore_weight = 1.0 / (1.0 + np.exp(z_for_weight))
    
    # Exploration term: UCB-like but scaled by weight
    # We use sigma / cost as the exploration signal
    explore_term = (sigma_safe / cost_safe) * explore_weight
    
    # Combine: Base EI/cost + Explore Term
    # We need to balance the scales. EI is in objective units. Sigma is in objective units.
    # Cost is in [1, 2] roughly.
    # So EI/cost and Sigma/cost are comparable in magnitude.
    
    scores = base_score + explore_term
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
