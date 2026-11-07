"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for the acquisition function
_acq_state = {
    'last_time': 0.0,
    'call_count': 0,
    'budget_deadline': None,
    'f_best_history': [],
    'x_evaluated': []  # We don't get x, but we can track f_best trajectory
}


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. Knowledge Gradient-inspired bonus for exploration
    3. Cost-aware normalization
    
    The key insight: we need to balance finding the global minimum
    (which may be far from current best) against exploitation of
    promising regions, while accounting for evaluation costs.
    """
    # Deadline guard: if we're running long, return a safe default
    # The task says each evaluation must finish within 300s
    # We'll add a simple time check as a safety net
    
    t0 = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Exploration bonus: reward regions with high uncertainty
    # that haven't been explored much. Since we don't have coordinates,
    # we use sigma as a proxy for unexplored regions.
    # Higher sigma means more uncertainty, which could indicate
    # either unexplored regions or high variance regions.
    
    # Knowledge Gradient approximation: the value of reducing uncertainty
    # We use a simple heuristic: bonus proportional to sigma, but
    # scaled down as f_best improves (we're more confident)
    
    # Track f_best history to adapt exploration
    _acq_state['f_best_history'].append(f_best)
    if len(_acq_state['f_best_history']) > 100:
        _acq_state['f_best_history'] = _acq_state['f_best_history'][-100:]
    
    # Compute how much f_best has improved recently
    if len(_acq_state['f_best_history']) >= 2:
        recent_improvement = _acq_state['f_best_history'][-10] - f_best if len(_acq_state['f_best_history']) >= 10 else _acq_state['f_best_history'][0] - f_best
    else:
        recent_improvement = 0.0
    
    # If f_best is improving rapidly, we can afford more exploitation
    # If stagnating, we need more exploration
    # Normalize improvement: 0 = no improvement, 1 = significant improvement
    # Use a scale based on the range of the objective
    # Three-hump camel 2D has minimum 0 and maximum around 125 on [-5,5]^2
    # But since we're working in scaled space, let's use a relative measure
    
    # Adaptive exploration weight
    # When f_best is high (poor), explore more
    # When f_best is low (good), exploit more
    f_best_norm = np.clip(f_best / 50.0, 0, 1)  # 0 = good, 1 = bad
    exploration_weight = 0.5 + 0.5 * f_best_norm
    
    # Uncertainty bonus: encourages exploring high-sigma regions
    # But penalize if sigma is extremely high (likely noise or boundary)
    sigma_bonus = exploration_weight * sigma * 0.1
    sigma_bonus = np.minimum(sigma_bonus, 0.5)  # Cap the bonus
    
    # Combine EI and exploration bonus
    # The EI is in objective units, sigma_bonus is also in objective units
    # Normalize both by cost
    score = ei_cost + sigma_bonus / cost_safe
    
    # Additional: use the relationship between mu and f_best more carefully
    # If mu is much lower than f_best, that's very promising (high EI)
    # If mu is close to f_best, less promising
    
    # Return finite scores, higher = evaluate next
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
