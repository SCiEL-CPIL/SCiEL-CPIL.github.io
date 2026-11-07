import numpy as np
import time

# Global state for timing and caching
_acq_state = {
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Minimization problem: Three Hump Camel 2D on [-5, 5]^2.
    """
    # Deadline guard
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:
        # Fallback: simple LCB
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        score = -(mu + 2.0 * s) / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Compute total cost to determine budget progress
    # The cost array contains the cost of each candidate point
    # Total budget is 30
    total_cost = np.sum(cost)
    budget = 30.0
    progress = min(total_cost / budget, 1.0)
    
    # Adaptive exploration parameter
    # Early on: more exploration (higher beta)
    # Late: more exploitation (lower beta)
    # Use a smooth decay
    beta = 2.0 * (1.0 - progress) + 0.5
    
    # Cost-aware UCB for minimization
    # We want to maximize: -mu + beta * sigma, penalized by cost
    # But we need to be careful: cost is the cost of evaluating each point
    # Higher cost means we should prefer cheaper points
    
    # Strategy: Normalize by cost, but also consider the potential improvement
    # The expected improvement for minimization with GP:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we don't have access to norm.pdf/cdf easily, so use UCB
    
    # Cost-aware UCB:
    # Score = (-mu + beta * sigma) / cost^alpha
    # where alpha controls how strongly we penalize cost
    
    # Use alpha = 0.5 (sqrt penalty) for a balance
    alpha = 0.5
    scores = (-mu + beta * s) / (c ** alpha + eps)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
