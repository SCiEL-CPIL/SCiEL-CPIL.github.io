"""Cost-aware acquisition: set-based greedy diversity-aware EI scoring.
Structural change: replace pointwise EI/cost with a batch-aware scoring
that penalizes spatial redundancy among high-scoring candidates. We
approximate the joint expected improvement by computing a diversity
penalty based on how many other candidates are "similar" (high mu,
low sigma) to each candidate, then combine with cost-aware EI.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep EI meaningful when GP is very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Compute base EI (expected improvement for minimization)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-aware EI: divide by cost
    ei_cost = ei / cost_safe

    # Set-based diversity penalty:
    # Approximate redundancy by looking at how many other candidates
    # have similar (mu, sigma) profiles. We use a simple heuristic:
    # for each candidate, count how many others have mu within a small
    # tolerance AND sigma within a small tolerance. Candidates in dense
    # regions get penalized.
    
    # To make this efficient, we use a grid-based approximation.
    # Bin mu and sigma into quantiles and count co-occurrences.
    n_bins = 20
    
    # Create bin indices for mu and sigma
    mu_quartiles = np.linspace(np.min(mu), np.max(mu) + 1e-12, n_bins + 1)
    sigma_quartiles = np.linspace(np.min(sigma_safe), np.max(sigma_safe) + 1e-12, n_bins + 1)
    
    mu_bin = np.digitize(mu, mu_quartiles) - 1
    mu_bin = np.clip(mu_bin, 0, n_bins - 1)
    sigma_bin = np.digitize(sigma_safe, sigma_quartiles) - 1
    sigma_bin = np.clip(sigma_bin, 0, n_bins - 1)
    
    # Count how many candidates share the same (mu_bin, sigma_bin) cell
    cell_key = mu_bin * n_bins + sigma_bin
    cell_counts = np.bincount(cell_key, minlength=n_bins * n_bins)
    redundancy = cell_counts[cell_key]
    
    # Normalize redundancy: map to [0, 1] with 1 = most redundant
    max_redundancy = max(float(np.max(redundancy)), 1.0)
    redundancy_norm = redundancy / max_redundancy
    
    # Diversity-aware score: reduce score for redundant candidates
    # Use a multiplicative penalty: score = ei_cost * (1 - alpha * redundancy_norm)
    # where alpha controls the strength of the diversity penalty
    alpha = 0.5
    diversity_factor = 1.0 - alpha * redundancy_norm
    diversity_factor = np.maximum(diversity_factor, 0.1)  # floor to avoid zeroing out
    
    score = ei_cost * diversity_factor
    
    # Add a small UCB component to ensure exploration even in low-EI regions
    ucb_component = (f_best - mu) + 2.0 * sigma_safe
    ucb_component = np.maximum(ucb_component, 0.0)
    ucb_component = ucb_component / cost_safe
    ucb_component = np.where(np.isfinite(ucb_component), ucb_component, 0.0)
    
    # Blend: primarily EI-cost-diversity, with a small UCB boost
    score = 0.8 * score + 0.2 * ucb_component * diversity_factor
    
    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
