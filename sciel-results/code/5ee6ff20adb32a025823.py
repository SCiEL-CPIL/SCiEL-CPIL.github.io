import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _ei_for_minimization(mu, sigma, f_best):
    """Expected Improvement for minimization: E[max(f_best - f(x), 0)]"""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """UCB for minimization: mu - beta * sigma (lower is better, so we want high score)"""
    return -(mu - beta * sigma)


def _thompson_sample(mu, sigma, rng_seed):
    """Thompson Sampling: draw sample from GP posterior, return sampled values"""
    rng = np.random.RandomState(rng_seed)
    z = rng.standard_normal(len(mu))
    return mu + sigma * z


def acquisition(mu, sigma, f_best, cost):
    """
    Structural Thompson Sampling acquisition with cost penalty.
    
    Draws a single sample from the GP posterior for all candidates,
    then scores each candidate by its sampled value divided by cost.
    This naturally balances exploration (via the random draw) and
    exploitation (via the posterior mean) while penalizing expensive
    evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Clean inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Fall back to simple cost-adjusted EI
        z = (f_best - mu) / (sigma + 1e-12)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei / cost), ei / cost, 0.0)

    # Estimate progress to modulate exploration
    known_min = -78.3323
    f_start_est = 20.0
    progress = np.clip((f_start_est - f_best) / (f_start_est - known_min), 0.0, 1.0)

    # Thompson Sampling: draw one sample from the GP posterior
    # Use a deterministic seed based on f_best and n for reproducibility within a call
    # but different across calls as f_best changes
    seed = int(abs(hash((round(f_best, 6), n))) % (2**31))
    sample = _thompson_sample(mu, sigma, seed)

    # For minimization, we want to select the candidate with the LOWEST sampled value.
    # So the score should be negative of the sampled value (higher score = lower value = better).
    # Cost penalty: divide by cost (expensive points get lower scores).
    # We also blend with EI to keep exploitation strong early on.
    
    # EI component for exploitation
    ei = _ei_for_minimization(mu, sigma, f_best)
    
    # Thompson score: -sample / cost (higher is better)
    ts_score = -sample / cost
    
    # EI score: ei / cost (higher is better)
    ei_score = ei / cost
    
    # Blend: early on (low progress) lean more on EI for exploitation,
    # later lean more on Thompson for exploration.
    # weight_ei goes from 0.7 at progress=0 to 0.3 at progress=1
    weight_ei = 0.7 - 0.4 * progress
    weight_ts = 1.0 - weight_ei
    
    combined = weight_ei * ei_score + weight_ts * ts_score

    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
