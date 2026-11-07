import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Deadline guard: if we are close to the limit, return a safe deterministic fallback
    # to ensure we don't crash or timeout. Although LCB is fast, this is a safety net.
    if time.time() > _DEADLINE:
        # Return negative mean as a safe greedy fallback
        return -mu
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization:
    # We want to minimize f(x).
    # The LCB estimate is mu - beta * sigma.
    # We want to select the point with the lowest LCB.
    # Since the acquisition function returns scores where HIGHER is better,
    # we return the negative of the LCB.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Adaptive beta: 
    # Early on, we want more exploration (higher beta).
    # Later, we want more exploitation (lower beta).
    # However, without knowing the exact step count easily inside this function 
    # (though we can infer it from f_best or just use a fixed value), 
    # a fixed beta is often robust. 
    # For Weierstrass, which is rough, a moderate exploration constant is good.
    # Standard LCB uses beta = 2 or 3. Let's use 2.0.
    beta = 2.0
    
    lcb = mu - beta * sigma
    
    # We want the minimum of lcb.
    # The evaluator picks the argmax of the returned scores.
    # So we return -lcb.
    scores = -lcb
    
    return scores
