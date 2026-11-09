import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # seconds, evaluator kills at 330s, keep 5s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Combines Expected Improvement (EI) with a Lower Confidence Bound (LCB) term.
    EI provides exploitation signal, while LCB provides linear exploration pressure
    to systematically sweep multimodal basins without premature convergence.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard: if we're getting close to the limit, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Simple fallback: just use -mu (exploitation only)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB term for minimization: mu - beta * sigma
    # We want to maximize this, so we negate: -mu + beta * sigma
    # This provides linear exploration pressure
    beta = 1.0
    lcb_score = -mu + beta * sigma
    
    # Combine EI and LCB
    # EI is in objective units, LCB is in objective units.
    # Weight them to balance exploitation and exploration.
    # For Michalewicz 2D (multimodal), we want moderate exploration.
    w_ei = 0.7
    w_lcb = 0.3
    
    scores = w_ei * ei + w_lcb * lcb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
