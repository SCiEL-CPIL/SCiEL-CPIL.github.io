"""Cost-aware acquisition for Bayesian optimization on Styblinski-Tang 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s margin before 300s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions
    sigma : array_like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-12)
    
    # Check deadline
    if _check_deadline():
        # Fallback: Exploit best (lowest mu) with cost penalty
        # We want to minimize f(x), so we maximize -f(x)/cost(x)
        # Approximate f(x) with mu(x)
        return -mu / cost

    # Thompson Sampling:
    # Sample a function value y for each candidate from the GP posterior
    # y ~ N(mu, sigma^2)
    # We want to find the candidate that minimizes y.
    # To incorporate cost, we maximize -y / cost.
    # This effectively samples a "cost-adjusted" function and picks the best.
    
    # Use a fixed seed for reproducibility? No, TS should be stochastic.
    # But numpy's random state is global. We can use np.random.
    # To be safe and avoid issues with global state, we can use a local RNG if needed, 
    # but np.random is standard.
    
    # Generate samples
    # y = mu + sigma * Z, where Z ~ N(0, 1)
    z = np.random.randn(mu.shape)
    y = mu + sigma * z
    
    # Score: We want to maximize the likelihood of being the best, adjusted for cost.
    # In pure TS, we pick argmin(y).
    # With cost, we pick argmin(y / cost)? Or argmin(y - lambda * cost)?
    # The prompt says "Cost-aware". The standard metric is "bang for buck".
    # If we minimize y, we get the best value.
    # If we minimize y/cost, we get the best value per cost.
    # Let's try maximizing -y / cost.
    
    scores = -y / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
