import numpy as np
from scipy.stats import norm
import time

# Cost-aware acquisition function for Bayesian optimization (levy 2d)
# Uses a structural change: instead of simple EI/cost, we use a
# "probability of improvement weighted by cost efficiency" combined with
# an uncertainty-aware term that adapts based on how much we've explored.
#
# Key structural difference from the parent:
# 1. We model the acquisition as a two-stage decision:
#    - First, how likely is this point to improve? (probability of improvement)
#    - Second, what's the expected gain per unit cost?
# 2. We add a "coverage" bonus that encourages visiting regions with high
#    posterior uncertainty, but scaled by the cost to evaluate there.
# 3. We use a hyperbolic discount on cost rather than linear division,
#    which gives a smoother trade-off.

# Track state across calls
_state = {
    'call_count': 0,
    'f_best_history': [],
    'total_cost_seen': 0.0,
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability-of-improvement
    combined with cost-efficient expected improvement and a coverage bonus.
    """
    _state['call_count'] += 1
    n_calls = _state['call_count']
    
    # Update state
    _state['f_best_history'].append(f_best)
    _state['total_cost_seen'] = float(np.mean(cost))
    
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip to valid ranges
    sigma_safe = np.clip(sigma, 1e-9, None)
    cost_safe = np.clip(cost, 1e-9, None)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma_safe * phi_z
    
    # Cost efficiency: use hyperbolic discount rather than linear division
    # This gives a smoother trade-off and avoids extreme values
    # cost_factor = 1 / (1 + 0.5 * cost_safe)  # hyperbolic
    # But let's try a different structural approach:
    # Weight EI by the "affordability" of the point
    
    # Structural change: use a softmax-like weighting over cost
    # Points with lower cost get higher weight, but not linearly
    cost_norm = cost_safe / (np.mean(cost_safe) + 1e-9)
    cost_weight = 1.0 / (1.0 + 0.3 * cost_norm)
    
    # Expected improvement per unit cost (smoothed)
    ei_cost = ei * cost_weight
    
    # Coverage bonus: encourage exploring high-uncertainty regions
    # Scale by how much we've explored so far
    # Early on, more exploration; later, more exploitation
    exploration_rate = max(0.05, 1.0 / (1.0 + 0.02 * n_calls))
    
    # Uncertainty-based coverage: sigma itself is a good proxy
    # But we want to avoid double-counting with EI
    # Use the part of sigma that isn't already captured by EI
    # This is tricky, so let's use a simpler approach:
    # Coverage = sigma * (1 - pi)  # high uncertainty AND low probability of improvement
    coverage = sigma_safe * (1.0 - pi) * exploration_rate
    
    # Combine: main signal is cost-efficient EI, secondary is coverage
    score = ei_cost + coverage
    
    # Add a small random tie-breaker to avoid plateaus
    # But keep it deterministic based on index for reproducibility
    # Actually, let's not add randomness - keep it clean
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize to [0, 1] for stability
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max - score_min > 1e-12:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.ones_like(score) / len(score)
    
    return score
