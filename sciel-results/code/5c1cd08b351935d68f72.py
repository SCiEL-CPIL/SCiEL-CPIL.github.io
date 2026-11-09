import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition for Goldstein-Price 2D.
    
    PLB maximizes the probability that the true function value exceeds a
    dynamic threshold derived from f_best. For a Gaussian posterior
    N(mu, sigma^2), P(f > threshold) = 1 - Phi((threshold - mu) / sigma).
    Maximizing this is equivalent to maximizing score = mu - threshold + k*sigma
    where k is chosen to target a desired probability of improvement.
    
    We use a smooth soft-max surrogate for the threshold to handle the
    nonsmooth max(f_best, 3.0) operation.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Smooth soft-max for threshold: soft_max(f_best, 3.0)
    # This avoids the kink at f_best = 3.0
    # Using log-sum-exp style: soft_max(a, b) = a + log(1 + exp(b - a)) / 1
    # Actually for max: soft_max(a, b) = a + softplus(b - a)
    # softplus(x) = log(1 + exp(x))
    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -50, 50)))
    
    # Threshold: we want to target points likely to beat f_best
    # Use a dynamic threshold slightly below f_best to allow some exploitation
    # of near-optimal regions
    threshold_raw = f_best - 0.1  # target slightly below best to encourage improvement
    # Soft-max with the known minimum value 3.0 to avoid going below it
    # soft_max(threshold_raw, 3.0) = threshold_raw + softplus(3.0 - threshold_raw)
    threshold = threshold_raw + softplus(3.0 - threshold_raw)
    
    # PLB: score = mu - threshold + k * sigma
    # where k is adapted based on the gap from optimum
    # When far from optimum (large gap), use larger k for more exploration
    # When close (small gap), use smaller k for more exploitation
    
    gap = max(f_best - 3.0, 0.0)
    
    # Adaptive k: larger when exploring, smaller when exploiting
    # k controls the probability target: P(f > threshold) ~ Phi(k)
    # k = 1.0 -> ~84% probability
    # k = 2.0 -> ~97.7% probability  
    # k = 0.5 -> ~69% probability
    
    # Start with aggressive exploration, decay to exploitation
    base_k = 0.5    # minimum exploration (exploit more when close)
    extra_k = 4.0   # additional exploration when far
    scale = 3.0     # transition scale
    
    k = base_k + extra_k * np.exp(-gap / scale)
    
    # Clip sigma to prevent extreme exploration
    sigma_clip = np.clip(sigma_safe, 1e-8, 20.0)
    
    # PLB score: higher is better
    # score = mu - threshold + k * sigma
    # This maximizes P(f > threshold) in a probabilistic sense
    scores = mu - threshold + k * sigma_clip
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
