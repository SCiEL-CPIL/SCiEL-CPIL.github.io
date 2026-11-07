import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability-weighted improvement.
    Maximizes P(mu < f_best) * (f_best - mu) / cost.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:  # Safety margin before 330s kill
        # Fallback: simple heuristic preferring low mu and low cost
        mu_fallback = np.asarray(mu, dtype=np.float64)
        cost_fallback = np.asarray(cost, dtype=np.float64)
        score = -mu_fallback / np.maximum(cost_fallback, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0. We want P(mu < f_best).
    # Let Y ~ N(mu, sigma^2). P(Y < f_best) = Phi((f_best - mu) / sigma).
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf
    # For very large positive z, Phi(z) -> 1
    # For very large negative z, Phi(z) -> 0
    z_clipped = np.clip(z, -50.0, 50.0)
    
    prob_improve = norm.cdf(z_clipped)
    
    # Expected improvement magnitude (ignoring the sigma*phi term for this specific heuristic)
    # This is the "mean" improvement conditioned on the model, weighted by probability.
    # Actually, (f_best - mu) is the deterministic gap.
    # The product prob_improve * (f_best - mu) is not exactly the expectation of the improvement,
    # but it is a valid heuristic that emphasizes points where the mean is already good 
    # and the probability of being better is high.
    
    # However, if mu is slightly above f_best but sigma is huge, prob_improve is ~0.5, 
    # and (f_best - mu) is negative. The score will be negative/small.
    # If mu is slightly below f_best, prob_improve is ~1, and (f_best - mu) is positive.
    
    # Let's ensure we only consider positive improvements
    improvement_gap = f_best - mu
    # If improvement_gap is negative, the score should be 0 (or very small)
    # prob_improve will be small, but let's be safe.
    
    score = prob_improve * improvement_gap
    score = np.maximum(score, 0.0) # Avoid negative scores for points unlikely to improve
    
    # Divide by cost
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
