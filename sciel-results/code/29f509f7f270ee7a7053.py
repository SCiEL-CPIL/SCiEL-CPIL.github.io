import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small exploration bonus.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: return -mu (greedy)
        return -mu
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Standard Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # To avoid underflow/overflow, we compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z) using scipy
    from scipy.stats import norm
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: when z is very negative, (f_best - mu) is very negative, but Phi(z) is near 0
    # When z is very positive, (f_best - mu) is very positive, Phi(z) is near 1
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Add a small exploration bonus based on sigma to encourage exploration early on
    # The bonus should be small enough not to dominate but large enough to help escape local optima
    # We scale the bonus by a factor that depends on how many evaluations have been done
    # Since we don't have the evaluation count, we use a fixed small coefficient
    exploration_bonus = 0.01 * sigma
    
    # Score: higher is better
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
