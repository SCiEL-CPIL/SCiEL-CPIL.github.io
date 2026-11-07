"""Cost-aware acquisition for Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start_time = time.time()
_DEADLINE_MARGIN = 25  # seconds before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _module_start_time
    if elapsed > 305 - _DEADLINE_MARGIN:
        # Fallback: simple LCB
        sigma_safe = np.maximum(sigma, 1e-8)
        return mu - sigma_safe

    sigma_safe = np.maximum(sigma, 1e-8)

    # Total cost consumed so far
    total_cost = float(np.sum(cost)) if len(cost) > 0 else 5.0
    # Budget is 30, so fraction consumed
    budget_frac = np.clip(total_cost / 30.0, 0.0, 1.0)

    # Adaptive beta for UCB: inversely proportional to budget consumption
    # Early (budget_frac ~ 0.17 after 5 points): beta is large -> exploration
    # Late (budget_frac ~ 1.0): beta is small -> exploitation
    # beta = 1 / (1 + total_cost / 30)
    beta_ucb = 1.0 / (1.0 + total_cost / 30.0)
    
    # Scale beta to reasonable range. 
    # At start (total_cost=5): beta = 1/(1+5/30) = 1/1.167 ≈ 0.857
    # At end (total_cost=30): beta = 1/(1+30/30) = 0.5
    # This is too narrow. Let's amplify the exploration in early stages.
    # Use a power law or a different scaling.
    # Let's try: beta = (1 - budget_frac) * 1.5 + 0.1
    # Start: 1.5*0.83 + 0.1 = 1.345
    # End: 1.5*0 + 0.1 = 0.1
    # This gives a stronger shift.
    beta_ucb = 0.1 + 1.4 * (1.0 - budget_frac)

    # Cost-aware UCB
    # score = -mu + beta_ucb * sigma
    # But we need to penalize cost.
    # Cost is in [1, 2]. Let's normalize cost to [0, 1] for penalty.
    cost_norm = (cost - 1.0) / 1.0  # [0, 1]
    
    # Cost penalty: divide by (1 + alpha * cost_norm)
    # Alpha should also adapt. Early: low alpha (explore even if expensive).
    # Late: high alpha (avoid expensive points).
    alpha_cost = 0.5 + 2.0 * budget_frac
    
    cost_penalty = 1.0 + alpha_cost * cost_norm
    
    # UCB score (for minimization, we want low mu, high sigma)
    # Standard UCB for minimization: -mu + beta * sigma
    ucb_score = -mu + beta_ucb * sigma_safe
    
    # Apply cost penalty
    # We want to reduce the score of expensive points.
    # Dividing by cost_penalty reduces the score.
    # But if ucb_score is negative, dividing by >1 makes it more negative (worse).
    # If ucb_score is positive, dividing by >1 makes it smaller (worse).
    # This seems correct.
    score = ucb_score / cost_penalty

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
