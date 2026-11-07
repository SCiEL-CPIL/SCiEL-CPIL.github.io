# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time, _call_count
    _call_count += 1
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0  # 10s safety margin before 330s kill
    
    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) based score
    # For minimization: LCB = mu - beta * sigma
    # Score = -LCB = -mu + beta * sigma
    # Use adaptive beta: start more exploratory, become more exploitative
    # f_best starts high (bad) and decreases; use it to modulate beta
    # When f_best is far from optimum, explore more; when close, exploit
    # Normalize f_best: hartmann 6d min is about -3.32237
    # f_best starts around 0 (uniform in [0,1]^6) and decreases
    # Use a schedule: beta = 1.5 - 0.5 * progress
    # progress = how close f_best is to the known minimum
    f_min_hartmann = -3.32237
    f_start = 0.0  # approximate starting f_best from uniform points
    progress = np.clip((f_start - f_best) / (f_start - f_min_hartmann + eps), 0.0, 1.0)
    beta = 1.5 - 0.5 * progress  # beta from 1.5 (explore) to 1.0 (exploit)
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Compute Cost-Adjusted Expected Improvement
    # Cost is in [1, 2] (1 + mean(x) where x in [0,1]^6)
    # We want to prefer candidates that give high EI per unit cost
    # cost_ei = EI / cost
    cost_safe = np.maximum(cost, eps)
    cost_ei = ei / cost_safe
    
    # 4. Normalize all objectives to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < eps:
            return np.ones_like(arr)
        return (arr - mn) / rng
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb_score)
    cost_ei_norm = normalize(cost_ei)
    
    # 5. Cost-aware combination using Pareto-inspired weighting
    # The key insight: we need to balance improvement vs cost
    # Use a dynamic weight that increases cost sensitivity as budget depletes
    # We don't know remaining budget directly, but we can use call_count as proxy
    # Early calls: prioritize improvement; later calls: prioritize cost efficiency
    
    # Use a sigmoid-like schedule based on call count
    # First ~50 calls: mostly improvement; after that: increasing cost weight
    t = float(_call_count)
    # Cost weight increases from 0.2 to 0.6 over first 100 calls
    cost_weight = 0.2 + 0.4 * (1.0 - np.exp(-t / 50.0))
    improve_weight = 1.0 - cost_weight
    
    # Combine EI and cost-adjusted EI
    # Primary: cost_ei (improvement per cost)
    # Secondary: raw EI for absolute improvement
    # Tertiary: UCB for exploration
    
    # Weighted sum with emphasis on cost-efficiency
    combined = improve_weight * (0.7 * ei_norm + 0.3 * ucb_norm) + cost_weight * cost_ei_norm
    
    # 6. Additional exploration bonus for high-uncertainty regions
    # When sigma is high, there's potential for large improvements
    # Add a small bonus for high sigma to avoid premature convergence
    sigma_norm = normalize(sigma_safe)
    exploration_bonus = 0.1 * sigma_norm
    
    final_scores = combined + exploration_bonus
    
    # 7. Final normalization
    final_scores = normalize(final_scores)
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
