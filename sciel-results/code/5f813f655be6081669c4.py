import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 300.0 # Wall clock limit

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_TIMEOUT - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian relaxation.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition values, shape (2000,), higher is better.
    """
    if not _elapsed_ok():
        return np.zeros_like(mu)

    # Lagrangian relaxation: score = (f_best - mu) - lambda * cost
    # We are minimizing f, so we want to maximize (f_best - mu) which is the expected improvement in value.
    # We penalize cost by lambda * cost.
    
    # Determine lambda.
    # The scale of f is ~10-100. The scale of cost is ~1-2.
    # If we set lambda = 0.1, a cost of 2 reduces the score by 0.2.
    # A potential improvement of 5.0 increases the score by 5.0.
    # This seems reasonable.
    
    # However, a purely mean-based score might lack exploration.
    # The directive specifies the form. I will stick to it but maybe add a tiny noise or use sigma if allowed?
    # "computing scores as f_best - mu - lambda * cost"
    # I will strictly follow this to test the structural change.
    
    # Adaptive lambda? 
    # If we are early, we might want to explore more (lower lambda).
    # If we are late, we might want to exploit cheap improvements (higher lambda).
    # Without knowing the step count, we can't easily adapt.
    # Let's use a fixed lambda that balances the scales.
    
    # Let's estimate the typical improvement per step.
    # If we assume we find the global min (-78) from start (0), total improvement is 78.
    # Over ~20 steps, that's ~4 per step.
    # Cost varies 1-2.
    # So lambda should be small enough to allow high-cost points if they offer big improvement.
    # lambda = 0.1 seems a bit low. lambda = 1.0?
    # If lambda = 1.0, a cost of 2 reduces score by 2.
    # An improvement of 2 is offset by the cost.
    # This forces the optimizer to find points that are significantly better *and* not too expensive.
    
    # Let's try lambda = 0.5.
    
    lambda_val = 0.5
    
    scores = (f_best - mu) - lambda_val * cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
