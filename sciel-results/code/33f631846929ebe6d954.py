import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Adaptive UCB
    # Estimate uncertainty scale
    sigma_max = np.max(sigma)
    sigma_med = np.median(sigma) + 1e-9
    
    # Beta adapts to the level of uncertainty
    # Early on, sigma is high, beta is high -> Exploration
    # Later, sigma is low, beta is low -> Exploitation
    # We want beta to be proportional to the "surprise" or relative uncertainty
    beta = 1.0 + (sigma_max / sigma_med)
    
    # Ensure beta doesn't get too large
    beta = np.clip(beta, 1.0, 5.0)
    
    # UCB Score: Maximize -mu + beta * sigma
    # We want to minimize f, so we maximize the "upper bound" of the potential minimum
    # Actually, for minimization, we use Lower Confidence Bound (LCB)
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB.
    # So Score = -LCB(x) = -mu(x) + beta * sigma(x)
    
    score = -mu + beta * sigma
    
    # Optional: Light cost penalty to respect the "cost-aware" nature
    # But keep it small so it doesn't dominate
    # cost is in [1, 2]
    # mu is in objective units (can be large)
    # So cost penalty must be scaled
    # If f ~ 100, cost 1 is negligible.
    # Let's add a small penalty to break ties or slightly prefer cheap points
    score -= 0.01 * cost
    
    return score
