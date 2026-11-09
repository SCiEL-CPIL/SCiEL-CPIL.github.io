"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blended EI + UCB strategy.
    
    - Rosenbrock 2D on [-2,2]^2, GP with Matern-5/2 kernel
    - Cost = 1 + mean(x_scaled) where x_scaled maps to [0,1]
    - Blend Expected Improvement per unit cost with UCB exploration
    - Adaptive kappa: higher exploration when far from optimum
    """
    t_start = time.time()
    
    # Safety deadline guard
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.clip(cost, 1.0, None)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # EI = delta * Phi(z) + sigma * phi(z)
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB score: mu - kappa * sigma (minimization)
    # Adaptive kappa: more exploration when f_best is large (early stage)
    # f_best starts high (~O(10) for rosenbrock) and approaches 0
    # Use log1p to get a smooth decay
    kappa_base = 1.5
    # Scale kappa down as we converge: when f_best is small, reduce exploration
    convergence_factor = np.exp(-np.log1p(np.maximum(f_best, 0.0) / 10.0))
    # When f_best is large, convergence_factor ~ 1 (full kappa)
    # When f_best is small, convergence_factor ~ 1 (still full kappa) 
    # Actually let's use a simpler approach: kappa decreases with f_best
    # kappa = kappa_base * (1 / (1 + f_best/5)) -- this reduces kappa as f_best grows? No...
    # We want MORE exploration early (large f_best) and LESS late (small f_best)
    # So kappa should be larger when f_best is large
    # kappa = kappa_base * (1 + f_best / 10)
    # But this could get very large. Let's cap it.
    kappa = kappa_base * (1.0 + np.minimum(f_best, 50.0) / 10.0)
    # Cap kappa to avoid excessive exploration
    kappa = np.minimum(kappa, 5.0)
    
    ucb = mu - kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is non-negative, UCB can be negative
    # Blend: 0.85 * (EI/cost) + 0.15 * (UCB/cost)
    # But UCB/cost can be negative, which is fine (penalizes bad points)
    
    ei_score = ei / cost_safe
    ucb_score = ucb / cost_safe
    
    # Blend with weights
    w_ei = 0.85
    w_ucb = 0.15
    
    score = w_ei * ei_score + w_ucb * ucb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 325:
        pass
    
    return score
# EVOLVE-BLOCK-END
