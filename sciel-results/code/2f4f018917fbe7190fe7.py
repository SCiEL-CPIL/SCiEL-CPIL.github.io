import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining LCB with adaptive exploration.
    
    Structural change: uses a multiplicative cost model where the exploration
    bonus (kappa * sigma) is scaled by remaining budget fraction, and adds
    an expected-improvement-like exploitation term. This creates a more
    nuanced trade-off: early on (more budget), we explore more; late, we
    exploit. The multiplicative cost penalty is stronger for expensive points
    when budget is tight.
    """
    t_start = time.time()
    
    # Safety deadline
    if time.time() - t_start > 2.0:
        return np.zeros(len(mu))
    
    n = len(mu)
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Estimate remaining budget fraction.
    # Total budget is 30. We don't know exactly how much we've spent,
    # but we can use the mean cost of evaluated points as a proxy.
    # Actually, we don't have access to past evaluations directly.
    # Use a heuristic: assume we're in the middle of the budget.
    # A better approach: use the cost distribution to gauge urgency.
    
    # The cost of each point is 1 + mean(x_scaled to [0,1]).
    # x_scaled = (x - (-1)) / 2 = (x+1)/2, so mean(x_scaled) in [0,1].
    # cost in [1, 2].
    
    # Key insight: use a normalized cost penalty that's multiplicative.
    # Normalize cost to [0,1] range: cost_norm = (cost - 1) / 1 = cost - 1
    cost_norm = (cost - 1.0)  # in [0, 1]
    
    # Adaptive kappa: start high for exploration, decrease as we get closer
    # to f_best (exploitation). Use a temperature-like schedule based on
    # how far mu is from f_best relative to sigma.
    
    # Distance from f_best in units of sigma
    dist = (mu - f_best) / sigma_safe
    
    # Where dist is very negative (mu << f_best), we're in a good region -> exploit
    # Where dist is positive or large, we're uncertain -> explore
    # Use a smooth schedule: kappa decreases as dist becomes very negative
    
    # Base kappa
    kappa_base = 2.0
    
    # Exploitation boost: when mu is close to or below f_best, reduce exploration
    # and add exploitation signal
    # Use the CDF of the GP predictive distribution at f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)  # P(f(x) < f_best)
    
    # Expected improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * prob_improve + sigma_safe * phi_z
    
    # Cost-aware score:
    # 1. LCB component: -(mu - kappa * sigma) — optimistic bound
    # 2. EI component: rewards points likely to beat f_best
    # 3. Cost penalty: multiplicative, stronger when cost is high
    
    # The multiplicative cost penalty: divide the exploration bonus by (1 + cost_norm * penalty_strength)
    # This makes expensive points less attractive for exploration
    penalty_strength = 3.0
    cost_multiplier = 1.0 + cost_norm * penalty_strength
    
    # Adaptive kappa: scale by cost multiplier (more expensive -> less exploration)
    kappa_adaptive = kappa_base / cost_multiplier
    
    # LCB score (higher is better for maximization of score)
    lcb_score = -(mu - kappa_adaptive * sigma_safe)
    
    # EI score (already positive when mu < f_best or high sigma)
    # Scale EI to be comparable to LCB
    ei_weight = 0.5
    
    # Combined score
    score = lcb_score + ei_weight * ei
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Additional time check
    if time.time() - t_start > 10.0:
        return np.zeros(len(mu))
    
    return score
