import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples a single latent function value from the posterior distribution
    at each candidate and returns these samples as scores.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Deadline guard: return a simple exploitation score if close to deadline
    if time.time() > _DEADLINE:
        # For minimization, we want low values, so return -mu as score
        # (higher score = better = lower actual value)
        return -mu

    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from N(mu, sigma^2) at each candidate
    # We want to MINIMIZE, so we should pick the candidate with the LOWEST sample
    # But the interface says "higher meaning evaluate next"
    # So we return negative of the sample (or just the sample and let the 
    # evaluator pick the max... wait, the evaluator picks the candidate with 
    # the HIGHEST score)
    
    # Since we want to minimize f, and Thompson sampling gives us a sample 
    # from the posterior, we should pick the point where the sample is lowest.
    # But the acquisition function must return scores where HIGHER = evaluate next.
    # So we return -sample, so that the minimum sample becomes the maximum score.
    
    # Actually, let's think about this more carefully.
    # Thompson sampling: sample f(x) ~ N(mu(x), sigma(x)^2)
    # We want to minimize f, so we pick argmin_x sample(x)
    # But the evaluator picks argmax_x score(x)
    # So we set score(x) = -sample(x)
    
    # Generate standard normal random numbers
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    
    # Sample from the posterior
    samples = mu + sigma_safe * z
    
    # Return negative samples so that lower f-values get higher scores
    scores = -samples
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
