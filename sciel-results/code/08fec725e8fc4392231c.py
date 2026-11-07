"""Cost-aware acquisition with Pareto-front hypervolume-inspired scoring for BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_start_time = None
_DEADLINE = 315.0  # safety margin before 330s kill


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        return 0.0
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-front inspired cost-aware acquisition: treats improvement magnitude
    and evaluation cost as two objectives, scoring candidates by their contribution
    to the hypervolume of the (improvement, 1/cost) objective space.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.ones(len(mu)) * 1e-10

    n = len(mu)
    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Objective 1: Expected Improvement (higher = better)
    z = (f_best - mu) / s
    z = np.clip(z, -12.0, 12.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Objective 2: Inverse cost (higher = cheaper = better)
    # Cost is in [1, 2] roughly, so 1/cost is in [0.5, 1]
    inv_cost = 1.0 / c

    # Normalize objectives to comparable scales for hypervolume calculation
    # EI scale: use max EI as reference
    ei_max = np.max(ei)
    if ei_max < eps:
        ei_max = 1.0
    ei_norm = ei / ei_max

    # Inv_cost is already in [0.5, 1], normalize to [0, 1]
    inv_cost_norm = (inv_cost - 0.5) / 0.5
    inv_cost_norm = np.clip(inv_cost_norm, 0.0, 1.0)

    # Hypervolume contribution approximation:
    # For a point (x, y) in 2D, its hypervolume contribution relative to
    # a reference point (0, 0) is x * y. But we want to account for
    # Pareto dominance: a point dominates another if it's better in both objectives.
    # 
    # Approximate hypervolume contribution as:
    # h(i) = ei_norm[i] * inv_cost_norm[i] * (1 + dominance_bonus[i])
    # where dominance_bonus reflects how many other points this one dominates
    #
    # For efficiency with 2000 points, we use a simpler approximation:
    # The Pareto front in 2D can be identified by sorting. Points on the
    # Pareto front have the property that no other point is better in both objectives.
    #
    # Simpler approach: use weighted sum with adaptive weights based on
    # the current Pareto front shape, plus a diversity bonus.
    
    # Adaptive weighting: when EI values are spread out, weight EI more;
    # when costs are spread out, weight cost more
    ei_spread = np.std(ei_norm) + eps
    cost_spread = np.std(inv_cost_norm) + eps
    
    # Weight that balances the two objectives
    w_ei = ei_spread / (ei_spread + cost_spread)
    w_cost = cost_spread / (ei_spread + cost_spread)

    # Weighted sum score
    weighted_score = w_ei * ei_norm + w_cost * inv_cost_norm

    # Add a Pareto front bonus: points closer to the ideal (1,1) corner get bonus
    # Distance from ideal in objective space
    dist_ideal = np.sqrt((1.0 - ei_norm)**2 + (1.0 - inv_cost_norm)**2)
    # Closer to ideal = smaller distance = higher bonus
    pareto_bonus = 1.0 / (1.0 + dist_ideal * 2.0)

    # Combine: weighted sum + Pareto bonus
    score = weighted_score * (0.7 + 0.3 * pareto_bonus)

    # Add uncertainty bonus: high sigma regions may contain better solutions
    # This encourages exploration in uncertain areas
    sigma_norm = s / (np.max(s) + eps)
    exploration_bonus = 0.1 * sigma_norm * inv_cost_norm

    score = score + exploration_bonus

    # Ensure finiteness and non-negativity
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    score = np.clip(score, 0.0, 1e6)

    return score
# EVOLVE-BLOCK-END
