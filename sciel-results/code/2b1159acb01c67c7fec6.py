"""Cost-aware PoI acquisition for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Probability of Improvement (PoI) acquisition function.
    
    score(x) = P(f(x) < f_best) / cost(x) + beta * sigma(x) / cost(x)
    
    PoI(x) = Phi((f_best - mu(x)) / sigma(x))
    
    This prioritizes candidates with high likelihood of improvement per unit
    of budget spent, rather than maximizing expected magnitude of improvement.
    The uncertainty bonus encourages exploration in poorly-known regions,
    also scaled by inverse cost.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t_start = time.time()
    deadline = 290.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    cost_safe = np.maximum(cost_safe, 1e-8)

    # Compute Probability of Improvement for minimization
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    improvement = f_best - mu
    sigma_for_poi = np.maximum(sigma_safe, 1e-10)
    
    z = improvement / sigma_for_poi
    z = np.clip(z, -40.0, 40.0)
    
    poi = norm.cdf(z)
    poi = np.clip(poi, 0.0, 1.0)
    poi = np.where(np.isfinite(poi), poi, 0.0)

    # Cost-adjusted PoI: prioritize high probability of improvement per unit cost
    poi_per_cost = poi / cost_safe

    # Auto-calibrate the exploration bonus weight
    # We want the exploration bonus to be a small perturbation relative to
    # the typical cost-adjusted PoI, so that PoI dominates but uncertainty
    # breaks ties and guides exploration.
    
    finite_poi_cost = poi_per_cost[np.isfinite(poi_per_cost) & (poi_per_cost > 1e-12)]
    if finite_poi_cost.size > 0:
        scale_poi_cost = float(np.median(finite_poi_cost))
    else:
        # Fallback: use a reasonable default scale
        scale_poi_cost = 0.1

    if not np.isfinite(scale_poi_cost) or scale_poi_cost < 1e-8:
        scale_poi_cost = 0.1

    finite_cost = cost_safe[np.isfinite(cost_safe)]
    scale_cost = float(np.median(finite_cost)) if finite_cost.size else 1.5
    if not np.isfinite(scale_cost) or scale_cost < 1e-8:
        scale_cost = 1.5

    finite_sig = sigma_safe[np.isfinite(sigma_safe)]
    scale_sigma = float(np.median(finite_sig)) if finite_sig.size else 0.1
    if not np.isfinite(scale_sigma) or scale_sigma < 1e-8:
        scale_sigma = 0.1

    # Set beta such that beta * scale_sigma / scale_cost = 0.15 * scale_poi_cost
    # This makes exploration a 15% tie-breaker factor relative to PoI/cost
    beta = 0.15 * scale_poi_cost * scale_cost / scale_sigma

    # Ensure beta is finite and non-negative
    beta = float(np.clip(beta, 0.0, 1e6))

    # Construct the final score
    # Higher score is better
    # PoI/cost is the primary driver
    # Uncertainty/cost is a secondary exploration bonus
    scores = poi_per_cost + beta * sigma_safe / cost_safe

    # Ensure all values are finite (higher = evaluate next)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard (safety margin before the 300s kill)
    if (time.time() - t_start) > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
