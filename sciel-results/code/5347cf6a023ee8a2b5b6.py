"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a structural change from EI/UCB blend to Probability of Improvement (PI)
    and UCB blend. PI is more robust for multimodal landscapes as it rewards
    the probability of finding *any* improvement, which helps escape local minima.
    The cost normalization is applied to both terms.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros(len(mu))

    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look for low mu.
    # UCB value = mu - beta * sigma
    # Score contribution = -UCB = -mu + beta * sigma
    # We want high scores, so we use -mu + beta*sigma
    
    # Adaptive beta and weights based on f_best
    # Early on (high f_best), explore more (higher beta, more UCB weight)
    # Late on (low f_best), exploit more (lower beta, more PI/EI weight)
    
    if f_best > 10.0:
        beta = 2.0
        w_pi = 0.4
        w_ucb = 0.6
    elif f_best > 5.0:
        beta = 1.5
        w_pi = 0.5
        w_ucb = 0.5
    elif f_best > 1.0:
        beta = 1.0
        w_pi = 0.7
        w_ucb = 0.3
    else:
        beta = 0.5
        w_pi = 0.8
        w_ucb = 0.2
        
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize UCB score to [0, 1] range roughly to be comparable with PI [0, 1]
    # We can use a soft normalization or just scale by a constant.
    # Since mu and sigma are in objective units, and PI is unitless,
    # we need to be careful.
    # Let's normalize ucb_score by the range of mu or a fixed scale.
    # A better way: use a sigmoid to map ucb_score to [0, 1]
    # Or simply divide by a typical scale of the objective function.
    # Rastrigin 2D min is 0, max is around 400+ (at corners).
    # Let's use a dynamic scale based on f_best or a fixed scale.
    # Using f_best as a scale is risky if f_best is 0.
    # Let's use a fixed scale of 10.0 for the objective function.
    scale = 10.0
    ucb_norm = np.tanh(ucb_score / scale)
    # tanh maps to [-1, 1]. Shift to [0, 1]
    ucb_norm = (ucb_norm + 1.0) / 2.0
    
    # Combine
    combined = w_pi * pi + w_ucb * ucb_norm
    
    # Cost-aware score
    scores = combined / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
