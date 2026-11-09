import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_acq_start_time = None
_ACQ_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with exponential cost suppression.
    
    Score = EI * exp(-k * cost)
    
    This combines the strong exploitation signal of Expected Improvement (EI) with a 
    multiplicative exponential cost gate. Unlike additive or linear division penalties, 
    the exponential gate sharply suppresses high-cost candidates unless the predicted 
    improvement is exceptionally large. This aligns with the finding that exponential 
    cost penalties force more aggressive exploitation of low-cost regions.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    global _acq_start_time
    
    # Initialize start time on first call
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    elapsed = time.time() - _acq_start_time
    if elapsed > 330.0 - _ACQ_DEADLINE_MARGIN:
        # If we are extremely close to timeout, return zeros to be safe
        # In practice, this vectorized computation is very fast, so this is unlikely to trigger
        return np.zeros(len(mu))
    
    # Prevent division by zero in z calculation
    # sigma should be > 0 due to noise=1e-6, but safeguard just in case
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # EI can be negative due to numerical errors or if logic is flipped, but theoretically >= 0.
    # Clip to 0 to ensure we don't "prefer" points with negative EI (which implies no improvement).
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Multiplicative Exponential Cost Gate
    # Instead of dividing by cost, we multiply by exp(-k * cost)
    # This creates a sharp penalty for high-cost points.
    # k is calibrated to the cost range [1, 2].
    # If k=1, cost=2 is penalized by exp(-2) ~ 0.135 relative to cost=1 (exp(-1) ~ 0.368).
    # Ratio is exp(1) ~ 2.7.
    # If k=2, ratio is exp(2) ~ 7.4.
    # We use k=1.5 for a strong but not prohibitive penalty, as suggested by the inspiration.
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    scores = ei * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
