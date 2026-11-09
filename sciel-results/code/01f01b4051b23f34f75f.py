import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return well before 300s
    # This is a single acquisition call, so it should be fast.
    # But we add a check just in case of pathological inputs.
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            # Fallback: return zeros if shape is wrong
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be huge. We clip to avoid numerical issues.
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: f_best is the best (lowest) value seen so far.
    # Since we are minimizing, improvement is f_best - mu (if mu < f_best).
    # The standard EI formula for minimization is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # This is positive when mu < f_best.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness
    # The cost is 1 + mean(x_scaled). x_scaled is in [0,1].
    # So cost is in [1, 2].
    # We want to prioritize points with low cost.
    # A common approach is to divide EI by cost, or use a penalty term.
    # The seed program used ei / cost and scored 1.0.
    # Inspiration 1 suggested adding a mild exploration term (sigma-based) to the numerator.
    # Let's try: (ei + beta * sigma) / cost
    # Where beta is a small constant to encourage exploration of uncertain regions.
    # However, since EI already incorporates sigma, adding sigma again might be redundant.
    # Alternatively, we can use a weighted combination of EI and UCB.
    # UCB = mu + kappa * sigma (for minimization, this is not quite right; UCB for minimization is mu - kappa * sigma? No, UCB is for maximization. For minimization, we want low mu and high sigma? No, high sigma means uncertainty, so we want to explore. But we also want to exploit low mu. So a good acquisition is mu - kappa * sigma? No, that's UCB for maximization. For minimization, we want to minimize. So we want low values. The acquisition function should be high when we want to evaluate. So we want high acquisition when mu is low or sigma is high.
    # Let's stick with EI/cost as it's proven.
    # Maybe we can tune the cost penalty.
    # cost_safe is in [1, 2].
    # If we use ei / cost_safe, points with cost=1 get 2x the score of points with cost=2 (for same EI).
    # This seems reasonable.
    
    # Let's try a slightly different approach:
    # Score = EI / (cost^alpha)
    # If alpha=1, it's the seed.
    # If alpha=0.5, it's softer.
    # If alpha=2, it's harder.
    # Let's try alpha=0.5 to see if it helps.
    
    # Actually, let's look at the inspiration: "add a mild exploration term (sigma-based) to the numerator"
    # scores = (ei + 0.1 * sigma) / cost
    # This adds a small boost to high-uncertainty points.
    
    # Let's try: scores = (ei + 0.05 * sigma_safe) / cost_safe
    # This is a mild change from the seed.
    
    scores = (ei + 0.05 * sigma_safe) / cost_safe
    
    # Check for non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf, return zeros to avoid invalid results
    if not np.any(np.isfinite(scores)):
        return np.zeros(n, dtype=np.float64)
    
    # Check time
    if time.time() - t0 > 280:
        # Return best effort
        return scores
    
    return scores
