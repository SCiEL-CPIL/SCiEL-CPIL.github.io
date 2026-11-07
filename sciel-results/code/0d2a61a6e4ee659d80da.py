import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Cost-aware weighting: PI^(1/cost)
    # This creates a multiplicative trade-off:
    # - High cost (expensive points) -> exponent near 0 -> score approaches 1 for any non-zero PI
    #   Wait, that's wrong. If cost is high, 1/cost is small, so PI^(small) is close to 1 even for small PI.
    #   That would favor expensive points!
    
    # Let's reconsider. We want to PENALIZE high cost.
    # If we use PI * (1/cost)^beta, then high cost reduces score.
    # If we use PI / cost^beta, same thing.
    
    # The issue with PI^(1/cost) is that it actually REWARDS high cost because
    # for a fixed PI in (0,1), PI^(1/cost) increases as cost increases.
    # Example: PI=0.5, cost=1 -> 0.5, cost=2 -> 0.707, cost=4 -> 0.84
    # So expensive points get higher scores, which is the opposite of what we want.
    
    # We need to INVERT the cost dependence. Let's use:
    # score = PI * (1 / cost^beta)
    # or score = PI / cost^beta
    
    # But we also want to keep the smooth, bounded nature.
    # Let's try: score = PI / (1 + cost - 1)^beta = PI / cost^beta
    # Since cost >= 1 (cost = 1 + mean(x_scaled) where x_scaled in [0,1], so cost in [1,2])
    
    # Let's use a moderate beta to balance exploration and cost avoidance.
    # beta = 1.0 seems reasonable: score = PI / cost
    
    # But this might be too aggressive. Let's try beta = 0.5: score = PI / sqrt(cost)
    # This is softer.
    
    cost_safe = np.maximum(cost, 1e-6)
    beta = 1.0
    
    score = pi / np.power(cost_safe, beta)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
