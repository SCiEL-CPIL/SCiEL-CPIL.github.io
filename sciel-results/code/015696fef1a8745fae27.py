import numpy as np
import time
from scipy.stats import norm

# Global state for timing and call counting
_acq_state = {
    'start_time': None,
    'call_count': 0
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Minimization problem: Three Hump Camel 2D on [-5, 5]^2.
    
    Uses a hybrid EI + UCB approach with adaptive weighting based on budget progress,
    combined with a cost-aware penalty that reflects the multi-objective trade-off
    between improvement magnitude and evaluation expense.
    """
    # Initialize state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    # Increment call count
    _acq_state['call_count'] += 1
    
    # Deadline guard - kill switch with safety margin
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 285:
        # Fallback: simple cost-aware EI
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        z = (f_best - mu) / s
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        ei = (f_best - mu) * Phi_z + s * phi_z
        ei = np.maximum(ei, 0.0)
        score = ei / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Calculate EI for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / s
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + s * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB score for minimization: -mu + beta * sigma
    # Adaptive beta: higher early for exploration, lower late for exploitation
    
    # Budget progress: estimate how much of the budget has been used
    # Cost per evaluation is 1 + mean(x_scaled_to_01), so cost is in [1, 2]
    # Total budget is 30, initial 5 points cost ~7.5, so remaining budget ~22.5
    # With avg cost ~1.5, we get ~15 more evaluations
    k = _acq_state['call_count']
    max_calls = 15.0
    budget_progress = min(k / max_calls, 1.0)
    
    # Adaptive UCB beta: decays from 1.5 to 0.8 as budget is consumed
    beta_ucb = 1.5 - 0.7 * budget_progress
    
    ucb_score = -mu + beta_ucb * s
    
    # Cost-aware penalty: instead of simple division by cost, use a softened penalty
    # that reflects the multi-objective trade-off.
    # Cost is in [1, 2]. We want to penalize expensive points but not too harshly.
    # Use: cost_penalty = 1 + 0.3 * (cost - 1)
    # This gives penalty in [1, 1.3], softer than 1/cost which gives [0.5, 1]
    cost_penalty = 1.0 + 0.3 * (c - 1.0)
    
    # Weighted combination of EI and UCB
    # Early: more UCB weight for exploration
    # Late: more EI weight for exploitation
    # EI weight increases from 0.3 to 0.8 as budget is consumed
    w_ei = 0.3 + 0.5 * budget_progress
    w_ucb = 1.0 - w_ei
    
    # Combine scores with cost penalty
    score_ei = ei / cost_penalty
    score_ucb = ucb_score / cost_penalty
    
    score = w_ei * score_ei + w_ucb * score_ucb
    
    # Add a small exploration floor for very uncertain, cheap regions
    # This helps escape local minima in early iterations
    exploration_weight = 0.2 * (1.0 - budget_progress)
    exploration_floor = exploration_weight * s / cost_penalty
    
    score = score + exploration_floor
    
    # Clean up - ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
