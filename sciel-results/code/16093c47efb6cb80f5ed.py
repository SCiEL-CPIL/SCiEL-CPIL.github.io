"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard (330s wall-clock limit, use 325s as safety margin)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop computing (near deadline)."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = better to evaluate next)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)  # avoid division by zero
    
    # Compute Expected Improvement (EI)
    delta = f_best - mu
    z = delta / sigma
    
    # Numerical stability for large |z|
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = delta * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we want low values
    # UCB = mu - beta * sigma (we prefer points with low mean and high uncertainty)
    # For minimization, the "upper" confidence bound in the sense of potential improvement
    # is actually mu - beta*sigma (lower prediction with uncertainty)
    # But for acquisition, we want to maximize the chance of finding a better point
    # So UCB for minimization: we want to minimize, so we look at mu - beta*sigma
    # The acquisition value should be higher when mu - beta*sigma is lower
    # So score_ucb = -(mu - beta*sigma) = -mu + beta*sigma
    # Or equivalently, we can use: beta*sigma - mu
    
    # Dynamic beta based on remaining budget
    # Total budget is 30, starting cost is about 5 (5 initial points)
    # So remaining budget ≈ 30 - current_total_cost
    # We estimate current total cost from the cost array and f_best
    # Actually, we don't have direct access to total cost, but we can infer
    # from the distribution of costs and how many evaluations have been done
    
    # A simpler approach: use the mean cost as a proxy for how much budget remains
    # Early on, costs are similar; later, the search may focus on cheaper regions
    # Let's use a fixed beta that balances exploration and exploitation
    
    # Use quantile-based gating: only consider candidates in the top 75% of cost-efficiency
    # This adapts to the distribution of costs
    
    # Compute cost-normalized EI
    ei_per_cost = ei / cost
    
    # Compute cost-normalized UCB-like score
    # For minimization: we want low mu and high sigma (uncertainty)
    # UCB_score = -mu / cost + beta * sigma / cost
    # But we need to choose beta carefully
    
    # Let's use an adaptive approach:
    # 1. Compute EI/cost and UCB/cost
    # 2. Blend them with a weight that adapts based on the spread of sigma
    
    # The spread of sigma indicates how much uncertainty remains
    # High spread -> more exploration needed -> weight UCB more
    # Low spread -> exploitation -> weight EI more
    
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    sigma_spread = sigma_std / (sigma_mean + 1e-9) if sigma_mean > 0 else 1.0
    
    # Clip spread to [0, 1] for blending
    sigma_spread = np.clip(sigma_spread, 0.0, 1.0)
    
    # Blend weight: when spread is high, use more UCB; when low, use more EI
    # w_ucb = 0.3 + 0.4 * sigma_spread
    w_ucb = 0.3 + 0.4 * sigma_spread
    w_ei = 1.0 - w_ucb
    
    # UCB for minimization: we want to maximize the probability of finding a value < f_best
    # A good proxy is: (f_best - mu) / cost + beta * sigma / cost
    # This is similar to EI but without the nonlinear transformation
    # Let's use: (f_best - mu + beta * sigma) / cost
    # where beta controls exploration
    
    # Adaptive beta: start with moderate exploration, decrease as we get closer to optimum
    # Use f_best as a proxy: if f_best is still large, explore more
    # levy 2d minimum is 0, so if f_best is close to 0, exploit more
    beta = 1.0 + 2.0 * np.clip(f_best / 10.0, 0.0, 1.0)
    
    # Compute UCB-like score
    ucb_score = (f_best - mu + beta * sigma) / cost
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # Combine EI and UCB
    score = w_ei * ei_per_cost + w_ucb * ucb_score
    
    # Apply quantile-based feasibility gate:
    # Keep only the top 75% of candidates by score, zero out the rest
    # This focuses the search on the most promising candidates
    if len(score) > 0:
        threshold = np.percentile(score, 25)  # 25th percentile
        score = np.where(score >= threshold, score, 0.0)
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
