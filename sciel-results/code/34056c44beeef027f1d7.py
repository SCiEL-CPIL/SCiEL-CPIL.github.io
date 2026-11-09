"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = time.time()
_DEADLINE_SECONDS = 330.0
_SAFETY_MARGIN = 5.0  # seconds

def _remaining_time():
    return _DEADLINE_SECONDS - (time.time() - _start_time) - _SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _probability_of_improvement(mu, sigma, f_best):
    """Probability of Improvement."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi

def _thompson_sample(mu, sigma, rng):
    """Thompson sampling: sample from posterior."""
    sigma_safe = np.maximum(sigma, 1e-8)
    samples = mu + sigma_safe * rng.standard_normal(size=mu.shape)
    return samples

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Uses a hybrid approach:
    1. Check if we're near deadline. If so, return simple EI/cost.
    2. Use Thompson Sampling with cost normalization.
    3. Add a small exploration bonus based on PI to escape local optima.
    
    Parameters:
    mu: (2000,) array of predicted means
    sigma: (2000,) array of predicted std deviations
    f_best: float, best observed value so far
    cost: (2000,) array of costs for each candidate
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: if time is running out, use fast deterministic method
    if _remaining_time() < 0.1:
        ei = _expected_improvement(mu, sigma, f_best)
        scores = ei / np.maximum(cost, 1e-8)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Thompson Sampling with cost-awareness
    # Use a fixed seed based on f_best to ensure reproducibility per state
    # but introduce stochasticity via the GP posterior
    rng = np.random.RandomState(int(abs(hash((f_best, len(mu)))) % 2**31))
    
    # Generate Thompson sample
    ts_sample = _thompson_sample(mu, sigma, rng)
    
    # Cost-normalized Thompson score
    # We want to maximize the sampled value (lower is better for objective, so we negate)
    # and divide by cost
    ts_score = -ts_sample / np.maximum(cost, 1e-8)
    
    # Add a Probability of Improvement component for exploration
    # PI gives a smooth gradient towards the valley floor
    pi = _probability_of_improvement(mu, sigma, f_best)
    
    # Blend Thompson score with PI-based exploration
    # The blend weight can be adaptive: more exploration when sigma is high
    sigma_mean = np.mean(sigma)
    # Higher sigma -> more weight on Thompson (exploration)
    # Lower sigma -> more weight on PI (exploitation)
    blend_weight = 0.5 + 0.5 * np.clip(sigma_mean / (sigma_mean + 0.1), 0, 1)
    
    # Normalize scores to comparable scales
    ts_score_norm = ts_score - np.min(ts_score)
    ts_score_norm = ts_score_norm / (np.max(ts_score_norm) + 1e-8)
    
    pi_norm = pi - np.min(pi)
    pi_norm = pi_norm / (np.max(pi_norm) + 1e-8)
    
    # Combined score
    scores = blend_weight * ts_score_norm + (1 - blend_weight) * pi_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
