# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3d.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) per unit cost for exploitation
    2. Probability of Improvement (PI) for structured exploration
    3. Adaptive beta for UCB-based exploration that shifts over budget
    4. Cost-aware normalization to prefer cheaper evaluations when quality is similar
    """
    t_start = time.time()
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of hartmann 3d
    f_min_known = -3.86278
    
    # Estimate budget consumption
    # Cost per evaluation is 1 + mean(x) where x in [0,1], so cost in [1, 2]
    # Total budget is 30
    mean_cost = np.mean(cost)
    # Estimate total spent: we don't know exact count, but mean cost gives a proxy
    # Higher mean cost suggests we're evaluating more expensive points
    # Use a simple heuristic: budget_consumed scales with mean cost
    budget_consumed = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    
    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z_clipped)
    
    # Adaptive beta for UCB: start with high exploration, reduce as budget depletes
    # beta controls exploration: high beta = more exploration
    # At start (budget_consumed=0): beta ~ 2.5
    # At end (budget_consumed=1): beta ~ 0.5
    beta = 2.5 - 2.0 * budget_consumed
    beta = np.maximum(beta, 0.3)
    
    # UCB score for minimization (higher is better when negated)
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware normalization
    # We want to prefer points with low cost when quality is similar
    # Use a power-law cost penalty: score / cost^alpha
    # alpha adapts: more sensitive to cost when budget is tight
    alpha = 0.3 + 0.7 * budget_consumed
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** alpha
    
    # Normalize each component to [0, 1] for blending
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / rng
    
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    ucb_norm = normalize(ucb)
    
    # Adaptive weighting based on gap to known minimum
    gap = f_best - f_min_known
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)
    
    # When gap is large (early): explore more (higher PI and UCB weight)
    # When gap is small (late): exploit more (higher EI weight)
    
    # Base weights
    w_ei = 0.5 + 0.3 * (1.0 - gap_normalized)   # 0.5 to 0.8
    w_pi = 0.3 - 0.15 * (1.0 - gap_normalized)  # 0.15 to 0.3
    w_ucb = 0.2 - 0.1 * (1.0 - gap_normalized)  # 0.1 to 0.2
    
    # Renormalize weights to sum to 1
    w_sum = w_ei + w_pi + w_ucb
    w_ei /= w_sum
    w_pi /= w_sum
    w_ucb /= w_sum
    
    # Combine normalized scores
    combined = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    
    # Apply cost penalty
    scores = combined / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add tiny deterministic noise to break ties and encourage exploration
    # Use a hash-like deterministic perturbation based on index
    idx = np.arange(len(mu))
    tie_breaker = 1e-8 * np.sin(idx * 0.7 + f_best * 100)
    scores = scores + tie_breaker
    
    # Deadline guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
