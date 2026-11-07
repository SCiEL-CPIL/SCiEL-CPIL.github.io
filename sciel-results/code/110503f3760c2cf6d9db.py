"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking and per-instance counters
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard: if we're getting close to the 330s wall-clock limit,
    # return a safe default to avoid timeout. Use a conservative margin.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 310.0:
        # Return a flat score to stop exploration safely
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure proper array types and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # For numerical stability, handle large |z|
    # For z > 8, Phi(z) ~ 1 and phi(z) ~ 0
    # For z < -8, Phi(z) ~ 0 and phi(z) ~ 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring with multiple components:
    
    # 1. Base cost-efficiency: EI per unit cost
    base_score = ei / cost
    
    # 2. Uncertainty-weighted cost efficiency:
    #    Prefer points that offer good EI relative to cost AND have reasonable uncertainty
    #    This helps balance exploitation and exploration
    #    Use a weighted combination that adapts based on how many evaluations we've done
    n_calls = _acq_state['call_count']
    
    # Early on, explore more (weight uncertainty higher)
    # Later, exploit more (weight improvement higher)
    # Total budget is ~30 cost, start with 5 initial points
    # Each additional point costs 1-2, so we can expect ~15-25 more evaluations
    # Let's say we expect about 20 more calls total
    
    if n_calls < 5:
        # Early phase: more exploration
        exploration_weight = 0.3
    elif n_calls < 15:
        # Middle phase: balanced
        exploration_weight = 0.15
    else:
        # Late phase: more exploitation
        exploration_weight = 0.05
    
    # Uncertainty component: sigma / cost (encourages evaluating uncertain points)
    # But scale it down so it doesn't dominate when EI is high
    uncertainty_score = sigma / cost
    uncertainty_score = np.minimum(uncertainty_score, 0.5 * np.max(uncertainty_score) + 1e-10)
    
    # 3. Cost normalization bonus:
    #    Add a small bonus for points with below-average cost
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-10
    cost_z = (cost - cost_mean) / cost_std
    # Points with lower cost get a slight bonus
    cost_bonus = 0.08 * (-cost_z)
    cost_bonus = np.clip(cost_bonus, -0.2, 0.2)
    
    # 4. Normalization-based approach inspired by inspiration 2:
    #    Normalize EI and cost to [0, 1] range for this batch
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Combine: favor high EI and low cost
    # score = ei_norm * (1 - cost_norm * gamma)
    gamma = 0.3  # Moderate cost penalty (reduced from 0.5 to be less aggressive)
    norm_score = ei_norm * (1.0 - gamma * cost_norm)
    
    # Blend the two approaches: weighted average of base_score and norm_score
    # This combines the absolute EI/cost signal with the relative normalization signal
    # The blend weight adapts: earlier calls favor the normalized approach (more exploration),
    # later calls favor the absolute approach (more exploitation)
    if n_calls < 8:
        blend_weight = 0.4  # More weight on normalized score early
    elif n_calls < 15:
        blend_weight = 0.3
    else:
        blend_weight = 0.2  # More weight on absolute score late
    
    combined_exploitation = (1.0 - blend_weight) * base_score + blend_weight * norm_score
    
    # Final combination:
    # Primary: blended exploitation score with cost bonus
    # Secondary: uncertainty for exploration, weighted
    scores = combined_exploitation * (1.0 + cost_bonus) + exploration_weight * uncertainty_score
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
