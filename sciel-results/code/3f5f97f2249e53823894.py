import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 5s margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    a lightweight Predictive Entropy Search (PES) approximation.
    
    For minimization:
    - EI favors points likely to improve upon f_best.
    - The PES-like term favors points that maximize the probability of
      finding a new global minimum (i.e., points where the posterior
      distribution has significant mass below f_best, especially in
      uncertain regions).
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard: if time is running out, fall back to a fast deterministic method
    if time.time() - _start_time > _TIME_LIMIT:
        # Fast fallback: use negative mean + uncertainty bonus
        sigma_safe = np.maximum(sigma, 1e-8)
        return -(mu + 0.5 * sigma_safe)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardize for EI calculation
    # EI for minimization: E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Phi and phi
    from scipy.stats import norm
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # PES-like term: approximate the mutual information between the function
    # value at x and the location of the global minimum.
    # A simple proxy: probability that f(x) < f_best, weighted by uncertainty.
    # This favors points where there's a good chance of beating f_best,
    # especially when uncertainty is high.
    prob_beat = Phi  # P(f(x) < f_best)
    
    # Combine EI and the PES-like term
    # The EI already captures most of the useful information, but we add
    # a small bonus for high uncertainty regions that still have a chance
    # of improvement, to encourage exploration of multimodal landscapes.
    # Weight: 0.8 * EI + 0.2 * (prob_beat * sigma_safe)
    # The second term rewards uncertainty in regions likely to be good.
    
    # Scale EI to be comparable with the second term
    # EI can be very small or zero for many points; normalize it
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    pes_term = prob_beat * sigma_safe
    pes_max = np.max(pes_term)
    if pes_max > 1e-10:
        pes_normalized = pes_term / pes_max
    else:
        pes_normalized = pes_term
    
    # Final score: higher is better
    # We want to maximize this score
    score = 0.8 * ei_normalized + 0.2 * pes_normalized
    
    # Add a small deterministic component to break ties and ensure
    # that points with lower mu are slightly preferred when other terms are equal
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-10:
        mu_normalized = (mu_max - mu) / (mu_max - mu_min)  # Higher for lower mu
    else:
        mu_normalized = np.zeros_like(mu)
    
    score = score + 0.05 * mu_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
