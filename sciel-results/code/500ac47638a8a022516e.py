"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Worst-Case Improvement (WCI) per unit cost with adaptive exploration.
    
    Uses a hybrid score that combines worst-case improvement with a
    normalized uncertainty term to balance exploitation and exploration.
    """
    # Adaptive exploration constant based on current best value
    # As f_best improves (becomes more negative), reduce exploration
    f_best_clamped = np.clip(f_best, -10.5, 10.0)
    progress = np.clip((0.0 - f_best_clamped) / 10.5, 0.0, 1.0)
    
    # Exploration constant: starts high, decreases as we find better solutions
    k_base = 2.0 - 1.5 * progress
    
    # Compute worst-case improvement (deterministic, no CDF/PDF)
    wci = np.maximum(0.0, f_best - mu - k_base * sigma)
    
    # Add a small normalized exploration bonus for high-uncertainty regions
    # This helps escape local minima in multimodal landscapes
    sigma_max = np.max(sigma) + 1e-9
    exploration_bonus = 0.1 * (sigma / sigma_max)
    
    # Combined score: WCI plus exploration bonus
    combined = wci + exploration_bonus
    
    # Guard against zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: combined improvement per unit cost (higher = evaluate next)
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
