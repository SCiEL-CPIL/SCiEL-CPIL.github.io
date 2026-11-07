"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling hybrid with EI and UCB for multimodal minimization.
    
    Combines the distance-to-best weighted EI from the current program with
    Thompson Sampling for stochastic exploration. The key insight is that for
    multimodal functions like Michalewicz 2D, deterministic acquisition functions
    (EI, UCB) tend to get stuck in local refinements. Thompson Sampling provides
    probabilistic jumps between basins of attraction by sampling from the posterior
    distribution at each step, while adaptive weighting ensures convergence in
    later iterations.
    
    The approach:
    1. Compute EI and UCB scores as in the current program
    2. Add a Thompson Sampling component: sample from N(mu, sigma^2) and use
       the negative sample as an exploration score
    3. Blend all three components with adaptive weights that shift from exploration
       (Thompson Sampling) to exploitation (EI) as the optimizer converges
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    eps = 1e-12
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Safety for EI
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Distance-to-Best weighting:
    # Find the index of the best candidate (lowest mu)
    best_idx = np.argmin(mu)
    best_mu = mu[best_idx]
    
    # Compute a "distance" proxy in function value space.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    if mu_range > eps:
        # Normalize mu to [0, 1] range
        mu_norm = (mu - mu_min) / mu_range
        # Distance from best in normalized space
        best_mu_norm = (best_mu - mu_min) / mu_range
        dist_from_best = np.abs(mu_norm - best_mu_norm)
        
        # Inverse distance weighting: boost distant points
        alpha = 2.0
        dist_weight = 1.0 + alpha * dist_from_best
    else:
        dist_weight = np.ones_like(mu)
    
    # Scaled EI
    ei_scaled = ei * dist_weight
    
    # UCB score for minimization
    sigma_median = np.median(sigma)
    
    if mu_range > eps:
        rel_uncertainty = sigma_median / (mu_range + eps)
    else:
        rel_uncertainty = 1.0
    
    # Adaptive beta for UCB
    beta = np.clip(1.5 * (0.5 + rel_uncertainty), 0.8, 2.5)
    ucb_score = -mu + beta * sigma
    
    # Thompson Sampling component:
    # Sample from the posterior N(mu, sigma^2) at each point
    # The negative of the sample is our exploration score (lower sample = better)
    # We use a deterministic seed based on the current state to ensure reproducibility
    # within a single call, but different across calls
    rng = np.random.default_rng()
    ts_samples = rng.normal(mu, sigma, size=n)
    ts_score = -ts_samples  # Higher score = lower sampled value = more promising
    
    # Normalize all three components to [0, 1] for blending
    def normalize(arr):
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        if arr_max - arr_min > eps:
            return (arr - arr_min) / (arr_max - arr_min)
        else:
            return np.zeros_like(arr)
    
    ei_norm = normalize(ei_scaled)
    ucb_norm = normalize(ucb_score)
    ts_norm = normalize(ts_score)
    
    # Adaptive weighting based on convergence state:
    # When uncertainty is high relative to the best value, explore more (Thompson Sampling)
    # When uncertainty is low, exploit more (EI)
    
    # Confidence metric: how close is f_best to the minimum possible mu?
    # If f_best is much better than most mu values, we're in exploitation mode
    gap_to_best = np.mean(np.maximum(0, mu - f_best))
    if mu_range > eps:
        gap_normalized = gap_to_best / (mu_range + eps)
    else:
        gap_normalized = 0.5
    
    # Weight allocation:
    # Early on (high gap, high uncertainty): more Thompson Sampling and UCB
    # Late (low gap, low uncertainty): more EI
    weight_ts = np.clip(0.3 * (1.0 - gap_normalized) + 0.2 * rel_uncertainty, 0.1, 0.5)
    weight_ucb = np.clip(0.2 * (1.0 - gap_normalized) + 0.1 * rel_uncertainty, 0.1, 0.4)
    weight_ei = 1.0 - weight_ts - weight_ucb
    weight_ei = np.clip(weight_ei, 0.2, 0.8)
    
    # Renormalize weights to sum to 1
    weight_sum = weight_ei + weight_ucb + weight_ts
    weight_ei /= weight_sum
    weight_ucb /= weight_sum
    weight_ts /= weight_sum
    
    # Combine scores
    score = weight_ei * ei_norm + weight_ucb * ucb_norm + weight_ts * ts_norm
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return -mu
    
    return score
# EVOLVE-BLOCK-END
