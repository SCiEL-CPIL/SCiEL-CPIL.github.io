"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale 2D function.
    
    This implementation uses a hybrid acquisition strategy that combines:
    1. Expected Improvement (EI) for exploitation of promising regions
    2. Multi-sample Thompson Sampling-inspired exploration for robustness
    3. A structured exploration term that balances uncertainty with improvement probability
    
    The Beale function is known for having multiple local minima, so pure EI
    can get stuck. This hybrid approach provides more robust exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Multi-sample optimistic Lower Confidence Bound (LCB) for exploration
    # This provides a more robust exploration signal than single-sample TS
    # by averaging multiple optimistic samples
    n_samples = 10
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility within a call
    
    # For each sample, draw from N(mu, sigma) and compute the optimistic LCB
    # The LCB is: -mu + beta * sigma (for minimization, we want low values)
    # But we want to maximize, so we use: mu - beta * sigma
    # Actually, for minimization with EI, we want to explore where f might be low.
    # Let's use a different approach: sample from the posterior and take the minimum
    # across samples, which gives an optimistic estimate.
    
    # Multi-sample optimistic estimate: for each point, draw n_samples from N(mu, sigma)
    # and take the minimum (optimistic for minimization)
    # This is equivalent to a multi-sample Thompson Sampling approach
    
    # To keep computation efficient, we'll use a vectorized approach
    # Generate all samples at once: shape (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, len(mu)))
    
    # Take the minimum across samples (optimistic for minimization)
    # This gives a lower estimate of the function value
    optimistic_estimate = np.min(samples, axis=0)
    
    # The exploration signal is how much lower this optimistic estimate is than the mean
    # If the optimistic estimate is much lower than mu, there's high uncertainty
    exploration_signal = mu - optimistic_estimate
    
    # Scale the exploration signal to be comparable to EI
    # The typical scale of exploration_signal is on the order of sigma * sqrt(2*log(n_samples))
    # Let's normalize by the mean sigma
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma > 0:
        exploration_signal = exploration_signal / mean_sigma
    
    # Combine EI with the exploration signal
    # The exploration weight should be moderate to balance exploitation and exploration
    # A weight of 0.3-0.5 is typical for this kind of hybrid
    exploration_weight = 0.4
    
    scores = ei + exploration_weight * exploration_signal
    
    # Add a small PI-weighted term for additional robustness
    # This term is large when both uncertainty is high and improvement is likely
    pi_weighted_exploration = 0.1 * pi * sigma_safe / (mean_sigma + 1e-9)
    
    scores = scores + pi_weighted_exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if computation took too long, return early
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
