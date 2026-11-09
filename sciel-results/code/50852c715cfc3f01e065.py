import numpy as np
from scipy.stats import norm
import time

# Module-level state to track budget consumption and timing
_acq_state = {
    'start_time': None,
    'total_cost_spent': 0.0,
    'f_best_history': [],
    'improvement_history': []
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy:
    1. Compute cost-normalized Expected Improvement (EI/cost) as the primary exploitation signal.
    2. Add a small, bounded exploration term (sigma/cost) to prevent premature convergence.
    3. Apply a Gaussian prior bias towards the expected cost of the minimizer (1.5 in scaled coords).
    4. Dynamically adjust the exploration weight based on remaining budget and recent improvement stagnation.
    5. Include a deadline guard to ensure safety within the 330s limit.
    """
    
    # Initialize timing
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 300:  # Safety margin before 330s limit
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
    
    # --- Compute Expected Improvement (EI) ---
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -1e4, 1e4)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # --- Exploration Term ---
    # Simple exploration: uncertainty per unit cost
    explore = sigma_safe / cost_safe
    
    # --- Dynamic Exploration Weight ---
    # Base weight is small (0.1) as in inspiration 1, but we modulate it based on budget and stagnation.
    budget_total = 30.0
    budget_spent = _acq_state['total_cost_spent']
    budget_remaining = max(budget_total - budget_spent, 1e-6)
    
    # Track f_best history
    if _acq_state['f_best_history']:
        prev_f_best = _acq_state['f_best_history'][-1]
        improvement = prev_f_best - f_best  # Positive means improvement
        _acq_state['improvement_history'].append(improvement)
    _acq_state['f_best_history'].append(f_best)
    
    # Determine exploration weight
    # Start with a moderate exploration weight
    beta = 0.1
    
    # If we have recent history, check for stagnation
    if len(_acq_state['improvement_history']) >= 3:
        recent_improvements = _acq_state['improvement_history'][-3:]
        mean_improvement = np.mean(recent_improvements)
        # If mean improvement is very small, we might be stuck -> increase exploration
        if mean_improvement < 0.01:
            beta = 0.2
    
    # If budget is low, reduce exploration to save cost for precise exploitation
    if budget_remaining < 5.0:
        beta = 0.05
    elif budget_remaining < 10.0:
        beta = 0.1
    
    # Combine EI and Exploration
    score = ei_cost + beta * explore
    
    # --- Prior Bias ---
    # The minimizer is in the inner 80% of [-4, 5]^4.
    # Scaled coordinates: x in [0, 1].
    # Original x in [-4, 5] -> x_scaled = (x + 4) / 9.
    # Inner 80% of original: [-3.1, 4.1].
    # Scaled inner region: [(-3.1+4)/9, (4.1+4)/9] = [0.1, 0.9].
    # Center of this region in scaled coords is 0.5.
    # Cost = 1 + mean(x_scaled).
    # Expected cost at minimizer = 1 + 0.5 = 1.5.
    
    target_cost = 1.5
    # Gaussian weight on cost deviation
    # Lambda controls the strength. 1.0 is moderate (from inspiration 1).
    lambda_bias = 1.0
    weight = np.exp(-lambda_bias * (cost_safe - target_cost) ** 2)
    
    score = score * weight
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update total cost spent
    # We estimate the cost of the next evaluation by taking the cost of the argmax score candidate.
    # This is an approximation, but it helps track the budget.
    if np.any(score > 0):
        best_idx = np.argmax(score)
        _acq_state['total_cost_spent'] += cost_safe[best_idx]
    
    return score
