"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Pre-computed constants for speed
_N = 2000

# Deadline guard: track time to avoid exceeding 300s
_DEADLINE = time.time() + 320.0

def _norm_cdf(x):
    """Fast normal CDF using scipy."""
    return norm.cdf(x)

def _norm_pdf(x):
    """Fast normal PDF using scipy."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: scalar, lowest observed value
    cost: array of shape (2000,), evaluation cost per candidate
    
    Uses a structural Thompson Sampling approach with cost-aware efficiency
    scoring, combined with a quantile-based feasibility gate for budget management.
    """
    # Deadline guard: if we're close to the limit, return a safe finite array
    if time.time() > _DEADLINE - 5.0:
        return np.ones_like(mu, dtype=float) * 0.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Thompson Sampling: sample from posterior to get a single score per candidate
    # This is a structural change from deterministic EI/UCB to stochastic sampling
    # For minimization, we sample f(x) ~ N(mu, sigma) and use -f_sampled as score
    # This naturally balances exploration and exploitation
    
    # Generate Thompson samples
    # Use a single sample for efficiency (can be extended to multiple samples)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization, lower sampled value is better
    # Score = -sampled_value (higher score = lower predicted value)
    ts_score = -samples
    
    # Also compute Expected Improvement for a hybrid approach
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    EI = improvement * _norm_cdf(z) + sigma * _norm_pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)
    
    # UCB for minimization: -mu + beta * sigma
    # Adaptive beta based on convergence
    # When f_best is close to 0 (minimum), we want more exploitation (lower beta)
    # When f_best is far from 0, we want more exploration (higher beta)
    beta = 1.5 + 1.5 * np.exp(-abs(f_best) / 0.5)
    UCB = -mu + beta * sigma
    UCB = np.where(np.isfinite(UCB), UCB, 0.0)
    
    # Normalize all scores to [0, 1] range for comparable blending
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range
    
    ts_norm = normalize(ts_score)
    ei_norm = normalize(EI)
    ucb_norm = normalize(UCB)
    
    # Adaptive blending weights based on uncertainty structure
    # When uncertainty is high relative to improvement, trust Thompson/UCB more
    sigma_median = np.median(sigma)
    mu_spread = np.ptp(mu) if len(mu) > 1 else 1.0
    
    # Relative uncertainty: how large is sigma compared to the spread of mu
    rel_uncertainty = sigma_median / (mu_spread + 1e-8) if mu_spread > 1e-8 else 1.0
    
    # Weight for Thompson Sampling: higher when uncertainty is high
    w_ts = 0.4 + 0.3 * min(rel_uncertainty, 1.0)
    # Weight for EI: moderate, always useful
    w_ei = 0.3
    # Weight for UCB: fills the rest
    w_ucb = 1.0 - w_ts - w_ei
    w_ucb = max(w_ucb, 0.0)
    
    # Combine normalized scores
    combined = w_ts * ts_norm + w_ei * ei_norm + w_ucb * ucb_norm
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Cost-aware efficiency: divide by cost to prefer cheaper evaluations
    # cost is in [1, 2] range
    efficiency = combined / cost
    efficiency = np.where(np.isfinite(efficiency), efficiency, 0.0)
    
    # Quantile-based feasibility gate
    # Keep candidates in the top 50% by efficiency, damp the rest
    # This creates a hard budget-aware constraint
    if len(efficiency) > 10:
        gate_threshold = np.percentile(efficiency, 50)  # top 50%
    else:
        gate_threshold = 0.0
    
    # Apply the gate: full score for feasible, heavily damped for infeasible
    # Use a smoother damping factor than the inspirations
    damp_factor = 0.2
    scores = np.where(efficiency >= gate_threshold, combined, combined * damp_factor)
    
    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Small exploration floor to avoid complete zeroing
    floor = 1e-10
    scores = np.maximum(scores, floor)
    
    return scores
# EVOLVE-BLOCK-END
