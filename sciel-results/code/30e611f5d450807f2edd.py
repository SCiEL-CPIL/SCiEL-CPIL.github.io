"""Cost-aware acquisition: phase-based EI + cost-normalized exploration."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def _ei(mu, sigma, f_best, eps=1e-9):
    """Standard Expected Improvement for minimization, with z-clipping for numerical safety."""
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, beta, eps=1e-9):
    """Upper Confidence Bound for minimization."""
    sigma_safe = np.maximum(sigma, eps)
    return mu - beta * sigma_safe

def _budget_phase_weight(total_cost, budget=30.0):
    """
    Returns (exploit_weight, explore_weight) based on budget phase.
    Early: more exploration. Late: more exploitation.
    """
    frac = np.clip(total_cost / budget, 0.0, 1.0)
    # Smooth transition: sigmoid centered at 0.5
    # At frac=0: exploit~0.1, explore~0.9
    # At frac=1: exploit~0.9, explore~0.1
    exploit_w = 0.5 + 0.5 * np.tanh(3.0 * (frac - 0.5))
    explore_w = 1.0 - exploit_w
    # Floor to ensure some of each
    exploit_w = np.maximum(exploit_w, 0.05)
    explore_w = np.maximum(explore_w, 0.05)
    total = exploit_w + explore_w
    return exploit_w / total, explore_w / total

def _cost_normalized_exploration(sigma, cost, eps=1e-9):
    """
    Exploration score: high uncertainty in cheap regions.
    Uses sigmoid(cost) to weight: cheap regions get more exploration boost.
    """
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    # Sigmoid: cheap (cost~1) -> ~0.73, expensive (cost~2) -> ~0.88
    # We want cheap regions to get MORE exploration, so invert
    cost_weight = 1.0 / (1.0 + np.exp(cost_safe - 1.5))
    return sigma_safe * cost_weight

def acquisition(mu, sigma, f_best, cost):
    """
    Phase-based cost-aware acquisition.
    Blends EI (exploitation) with cost-normalized uncertainty (exploration).
    """
    # Deadline guard: track time
    t_start = time.time()
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute EI
    ei = _ei(mu, sigma, f_best)
    
    # Compute exploration score
    exploration = _cost_normalized_exploration(sigma, cost_safe)
    
    # Get phase weights based on total cost
    total_cost = float(np.sum(cost_safe))
    exploit_w, explore_w = _budget_phase_weight(total_cost)
    
    # Normalize EI and exploration to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    exp_max = np.max(exploration) if np.max(exploration) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    exp_norm = exploration / exp_max
    
    # Combine: weighted blend, then divide by cost for cost-awareness
    combined = exploit_w * ei_norm + explore_w * exp_norm
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we're running long, return early
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score

# EVOLVE-BLOCK-END
