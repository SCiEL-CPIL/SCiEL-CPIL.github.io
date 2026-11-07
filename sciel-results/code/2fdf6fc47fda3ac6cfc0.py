# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses dynamically annealed UCB with soft Pareto boost, inspired by the
    best-performing solution (score 2.1538).
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, the lowest observed value so far
    - cost: array of shape (2000,) with evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the 330s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple fallback to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute budget progress: we don't know total budget exactly, but we can
    # estimate from the cost array. The total budget is ~30, starting cost is ~5
    # (5 initial points). We use the mean cost as a proxy for progress.
    # Actually, we don't have direct access to total cost spent. Let's use
    # a principled approach: estimate progress from f_best improvement potential.
    
    # Dynamically annealed UCB beta:
    # Start with high exploration, decay to exploitation as budget is used.
    # We estimate budget fraction used. The initial 5 points cost ~5*(1+0.5)=7.5
    # Total budget is 30, so remaining is ~22.5. Each new point costs ~1.5 avg.
    # We don't know exactly how many points have been evaluated, so we use
    # a heuristic: the spread of mu and sigma gives us a signal.
    
    # Better approach: use the relationship between f_best and mu to estimate
    # how "explored" the space is. If f_best is close to the minimum of mu,
    # we're more in exploitation mode.
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    # If mu_range is small, we've explored a lot -> more exploitation
    # If mu_range is large, we're early -> more exploration
    # Normalize: progress = 1 - mu_range / (mu_range + 1)
    # This gives 0 when mu_range is large, 1 when mu_range is small
    progress = 1.0 - mu_range / (mu_range + 1.0)
    
    # Anneal beta from 2.0 (exploration) to 0.5 (exploitation)
    beta = 2.0 - 1.5 * progress
    
    # UCB score: mu + beta * sigma
    ucb = mu + beta * sigma_safe
    
    # Also compute EI for a combined score
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Soft Pareto boost: combine improvement, uncertainty, and cost dimensions
    # Normalize each component to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(-ucb) if np.max(-ucb) > 0 else 1.0  # lower mu is better
    cost_max = np.max(cost_safe) if np.max(cost_safe) > 0 else 1.0
    
    # Normalize (lower is better for cost, so invert)
    ei_norm = ei / ei_max
    ucb_norm = (-ucb) / ucb_max  # higher is better (lower mu)
    cost_norm = 1.0 - (cost_safe / cost_max)  # higher is better (lower cost)
    
    # Weighted combination with soft Pareto
    # Weights: improvement 0.4, uncertainty 0.3, cost 0.3
    w_ei = 0.4
    w_ucb = 0.3
    w_cost = 0.3
    
    # Soft Pareto: use product of ranks to avoid dominance by one dimension
    # Rank-based approach for robustness
    ei_rank = np.argsort(np.argsort(ei)) / len(ei)
    ucb_rank = np.argsort(np.argsort(-ucb)) / len(ucb)
    cost_rank = np.argsort(np.argsort(-cost_safe)) / len(cost_safe)
    
    # Combined score: weighted sum of normalized ranks + raw EI/cost
    rank_score = w_ei * ei_rank + w_ucb * ucb_rank + w_cost * cost_rank
    
    # Primary score: EI per unit cost (proven effective)
    primary_score = ei / cost_safe
    
    # Secondary: UCB per unit cost
    secondary_score = (-ucb) / cost_safe
    
    # Blend: 60% EI/cost, 40% rank-based Pareto
    scores = 0.6 * (primary_score / (np.max(primary_score) + 1e-9)) + \
             0.4 * rank_score
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
