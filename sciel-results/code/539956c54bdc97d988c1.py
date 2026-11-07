"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using UCB on reward (negative expected regret)
    with dynamic beta scaled by remaining budget ratio.
    
    Strategy:
    1. Compute reward = -mu (higher is better for minimization)
    2. Apply UCB: reward + beta * sigma
    3. Divide by sqrt(cost) for cost-awareness
    4. Scale beta by remaining budget to shift exploration->exploitation
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Estimate remaining budget: total budget is 30, initial 5 points cost ~7.5
    # So remaining is roughly 30 - 7.5 = 22.5 at start
    # We don't know exact spent, so use f_best as proxy for progress
    # When f_best is large, we're early; when small, we're late
    
    # Budget ratio: 1.0 = full budget remaining, 0.0 = exhausted
    # Use f_best magnitude as proxy: large f_best = early, small = late
    # Three hump camel range: min 0, max ~-2.02 (at (-1.02, 2.02))
    # But we're minimizing, so f_best starts high and decreases
    # Normalize: assume initial f_best ~ 1-5, final ~ 0
    f_best_clipped = max(f_best, 0.0)
    # Budget ratio: starts near 1, decreases as f_best approaches 0
    # Use exponential decay for smooth transition
    budget_ratio = np.exp(-f_best_clipped / 2.0)  # 1 at f_best=0, ~0.6 at f_best=1
    
    # Beta: exploration coefficient
    # Start with high exploration, decrease as budget depletes
    # beta = base + scale * budget_ratio
    base_beta = 0.5
    scale_beta = 2.0
    beta = base_beta + scale_beta * budget_ratio
    
    # UCB on reward: -mu + beta * sigma
    reward = -mu
    ucb_reward = reward + beta * sigma_safe
    
    # Cost-awareness: divide by sqrt(cost)
    # cost is in [1, 2], so sqrt(cost) in [1, ~1.414]
    cost_safe = np.maximum(cost, 1.0)
    score = ucb_reward / np.sqrt(cost_safe)
    
    # Add a small EI component for exploitation when we're confident
    # This helps focus on regions where we expect improvement
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    from scipy.stats import norm
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Blend: more EI weight when budget is low (exploit), more UCB when high (explore)
    ei_weight = 0.3 * (1.0 - budget_ratio)  # 0 when budget high, 0.3 when low
    score = (1.0 - ei_weight) * score + ei_weight * EI / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
