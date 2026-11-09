import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute Upper Confidence Bound (UCB) scores for candidate points.
    Uses a time-decaying beta to balance exploration and exploitation.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        UCB scores for each candidate, shape (N,). Higher scores indicate better candidates.
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Determine current step based on how close f_best is to the known minimum
    # The task states the minimum value is -1.8013.
    # We can estimate the step by tracking how many evaluations have likely occurred,
    # but since we don't have a step counter passed in, we can infer progress from f_best.
    # However, a simpler robust heuristic is to use a fixed schedule based on the fact
    # that we have 30 evaluations total.
    # Since we cannot track state without a global variable (which is allowed but let's try to be stateless if possible, 
    # or use a simple global counter if needed).
    
    # The prompt says "Keep the function names and signatures of the seed program".
    # The seed program used a global state. I will use a simple global counter to track steps
    # to implement the time-decaying beta accurately, as f_best alone is a noisy proxy for time.
    
    # Global step counter
    if not hasattr(acquisition, 'step'):
        acquisition.step = 0
        
    acquisition.step += 1
    t = acquisition.step
    
    # Total evaluations is 30.
    # beta(t) = 2.0 * (1 - t/30)
    # At t=1, beta ~ 1.93 (high exploration)
    # At t=29, beta ~ 0.06 (high exploitation)
    # At t=30, beta = 0 (pure exploitation)
    
    beta = 2.0 * (1.0 - t / 30.0)
    
    # Ensure beta doesn't go negative
    beta = max(beta, 0.0)
    
    # UCB for minimization:
    # We want to find low values.
    # The "confidence bound" for the true value f(x) is roughly mu(x) - beta * sigma(x) for lower bound.
    # However, UCB acquisition typically maximizes: -mu(x) + beta * sigma(x)
    # This is equivalent to minimizing: mu(x) - beta * sigma(x)
    # Since we return scores where higher is better, we return:
    # score = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
