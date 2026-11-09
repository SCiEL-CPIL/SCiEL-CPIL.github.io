import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    Uses the standard Expected Improvement metric, normalized by the evaluation cost.
    This differs from LCB by considering the probability of improvement rather than just the bound.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Deadline check
    def time_left():
        return 330.0 - (time.time() - t0) - 5.0
    
    if time_left() <= 0:
        # Fallback: simple LCB if we are running out of time (EI is more expensive)
        kappa = 1.5
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) 
    #        + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # Avoid division by zero in z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in norm.cdf/pdf which can lead to numerical issues
    # Phi(z) -> 0 for z << 0, Phi(z) -> 1 for z >> 0
    # phi(z) -> 0 for |z| >> 0
    # We can safely clip to [-10, 10]
    z = np.clip(z, -10.0, 10.0)
    
    # Calculate EI components
    # Note: For minimization, improvement is f_best - f(x)
    # If mu < f_best, there is high probability of improvement.
    # The formula above is for maximization usually? Let's check.
    # Standard EI for minimization:
    # Let Y ~ N(mu, sigma^2). We want E[max(f_best - Y, 0)].
    # Let Z = (Y - mu)/sigma ~ N(0,1).
    # f_best - Y = f_best - (mu + sigma*Z) = (f_best - mu) - sigma*Z.
    # E[max((f_best - mu) - sigma*Z, 0)]
    # Let A = f_best - mu.
    # E[max(A - sigma*Z, 0)]
    # If A/sigma is large positive, Z needs to be very negative.
    # Integral of (A - sigma*z) * phi(z) dz from -inf to A/sigma.
    # = A * Phi(A/sigma) - sigma * [ -phi(z) ]_{-inf}^{A/sigma}
    # = A * Phi(A/sigma) + sigma * phi(A/sigma)
    
    # So EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
