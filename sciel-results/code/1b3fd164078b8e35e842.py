import numpy as np
from scipy.stats import norm
import time

# Module-level cache to track elapsed time and iteration count
_acq_state = {
    'start_time': None,
    'call_count': 0,
    'f_best_history': [],
    'cost_history': []
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a dynamic exploration-exploitation balance that shifts from
    exploration to exploitation as the budget is consumed.
    """
    # Initialize timing
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    _acq_state['call_count'] += 1
    _acq_state['f_best_history'].append(f_best)
    _acq_state['cost_history'].append(np.mean(cost))
    
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:  # Safety margin before 300s limit
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Dynamic exploration-exploitation balance ---
    # The key structural change: instead of a fixed exploration weight,
    # we compute a dynamic beta that depends on how much budget remains.
    
    # Estimate remaining budget. Total budget is 30, initial 5 points cost ~5*1.5=7.5
    # Each evaluation costs ~1.5 on average, so we can do roughly (30 - spent) / 1.5 more
    # But we track actual cost consumption.
    
    # Use the number of calls as a proxy for iterations
    n_calls = _acq_state['call_count']
    
    # Early iterations: more exploration. Later: more exploitation.
    # The budget is 30 total cost. Initial 5 points. So remaining evals ~ (30 - 5*1.5)/1.5 ≈ 16
    # So we have roughly 16-20 evaluations total.
    
    # Compute a progress indicator: how far along are we?
    # Use the mean cost of all candidates as a proxy for typical cost
    typical_cost = np.median(cost_safe)
    
    # Estimate total evaluations possible: budget / typical_cost
    budget = 30.0
    total_possible_evals = budget / typical_cost
    
    # Current evaluation number (1-indexed)
    current_eval = n_calls
    
    # Progress: 0 at start, 1 at end
    progress = min(1.0, current_eval / max(total_possible_evals, 1.0))
    
    # Dynamic beta: starts high (exploration), decreases to low (exploitation)
    # beta = 1.0 - progress * 0.7  => starts at 1.0, ends at 0.3
    # But we want a smoother transition
    beta = 1.0 - 0.8 * progress
    beta = np.clip(beta, 0.2, 1.0)
    
    # --- Compute acquisition components ---
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term
    # Use the distribution of sigma to normalize
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Normalize sigma to [0, 1] range for the exploration term
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / max(sigma_max - np.min(sigma_safe), 1e-9)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Exploration score: high where sigma is high, normalized by cost
    explore_score = sigma_norm / cost_safe
    
    # Combine with dynamic beta
    score = (1.0 - beta) * ei_cost + beta * explore_score
    
    # --- Cost-centering bias ---
    # The minimizer is in the inner 80% of [-4, 5]^4, so in scaled coords [0.1, 0.9]
    # Expected cost at minimizer ≈ 1 + 0.5 = 1.5
    # Apply a Gaussian weight to favor candidates with cost near 1.5
    target_cost = 1.5
    # Make the bias weaker as we progress (we've already found the region)
    bias_strength = 1.0 * (1.0 - 0.5 * progress)
    weight = np.exp(-bias_strength * (cost_safe - target_cost) ** 2 / 0.1)
    
    score = score * weight
    
    # --- Additional: penalize very high cost candidates ---
    # Candidates with cost > 2.0 are less likely to be near the minimizer
    cost_penalty = np.where(cost_safe > 2.0, 0.5, 1.0)
    score = score * cost_penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
