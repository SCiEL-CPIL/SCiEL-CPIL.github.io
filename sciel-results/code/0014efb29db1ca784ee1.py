"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track deadline across calls within a single evaluation run
_start_time = None
_DEADLINE_FRACTION = 0.95  # Safety margin: stop at 95% of 330s
_call_count = 0

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _reset_timer():
    global _start_time, _call_count
    _start_time = time.time()
    _call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Knowledge Gradient (KG) approximation.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP std predictions
    f_best: lowest observed value
    
    Returns: (2000,) array of scores, higher = more promising
    """
    global _start_time, _call_count
    
    # If this is the first call, reset the timer
    if _start_time is None:
        _reset_timer()
    
    _call_count += 1
    n_evals = _call_count + 4  # 5 initial points + calls so far
    
    elapsed = _get_elapsed()
    deadline = 330.0 * _DEADLINE_FRACTION
    
    # Safety guard: if we're close to deadline, return a simple fallback
    if elapsed > deadline:
        return -mu  # Exploit: pick the point with lowest predicted mean
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Knowledge Gradient approximation:
    # For each candidate i, the expected improvement in the minimum is:
    # E[max(f_best - min(mu_i, true_i), 0)]
    # A common approximation is to sum over all candidates the probability
    # that candidate i is the new best, weighted by its expected improvement.
    
    # First, compute the probability that each candidate is the best
    # among all candidates. This is approximated by:
    # P(candidate i is best) ≈ ∏_{j≠i} P(mu_i + noise_i < mu_j + noise_j)
    # For Gaussian, P(a < b) = Φ((mu_a - mu_b) / sqrt(sigma_a^2 + sigma_b^2))
    
    # But with 2000 candidates, this is O(n^2) which might be too slow.
    # Instead, use a simpler KG-like score that captures the global view:
    
    # The key insight of KG is that it values points that have a high
    # probability of being the global minimum, not just local improvement.
    
    # Approximation: score_i = E[f_best - min_i] where min_i is the minimum
    # over all candidates, but we approximate by considering only the top
    # candidates.
    
    # A practical approximation:
    # 1. Compute the "improvement potential" of each candidate: how much
    #    better it could be than f_best
    # 2. Weight by the probability that it's actually the best
    
    # Compute z-scores relative to f_best
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Probability that candidate i is better than f_best
    p_better = norm.cdf(z_clipped)
    
    # Expected improvement if evaluated
    ei = (f_best - mu) * p_better + sigma_safe * norm.pdf(z_clipped)
    
    # For KG, we want to estimate the expected reduction in the minimum
    # after evaluating one more point. This is approximately:
    # E[max(f_best - min(all candidates), 0)]
    
    # A simpler but effective approximation:
    # score_i = p_i * ei_i where p_i is the probability that i is the best
    # among all candidates.
    
    # To compute p_i efficiently, we can use the fact that for a set of
    # Gaussian random variables, the probability that i is the minimum
    # can be approximated.
    
    # A common practical approximation is to use:
    # p_i ∝ exp(-mu_i / temperature) * sigma_i
    # This favors points with low mean and high uncertainty.
    
    # But let's try a more principled approach:
    # The KG value is approximately the expected maximum improvement
    # over all candidates. We can approximate this by:
    
    # 1. Sort candidates by mu (ascending)
    # 2. For each candidate, compute the probability it's the best
    #    by integrating over the distribution of the minimum of the others.
    
    # For efficiency, use a simpler score that captures the KG spirit:
    # score_i = E[f_best - X_i] * P(X_i is the best)
    # where X_i ~ N(mu_i, sigma_i^2)
    
    # P(X_i is the best) ≈ P(X_i < X_j for all j≠i)
    # This is hard to compute exactly, but we can approximate.
    
    # A practical KG approximation:
    # score_i = (f_best - mu_i) * Φ((f_best - mu_i)/sigma_i) * 
    #           σ_i * φ((f_best - mu_i)/sigma_i) * 
    #           [1 + correction for being the global best]
    
    # Let's use a weighted combination that emphasizes global optimality:
    
    # Compute a "global best" indicator: points that are likely to be
    # the minimum over the entire set.
    
    # Method: For each point, compute the probability it's the minimum
    # by considering pairwise comparisons with the top few candidates.
    
    # For efficiency, use a simpler score:
    # score = EI * (1 + uncertainty bonus for global exploration)
    
    # The uncertainty bonus should be higher for points in regions
    # where the GP is uncertain, as these might hide the global minimum.
    
    # Adaptive exploration: early on, explore more; later, exploit.
    # Use the number of evaluations to schedule exploration.
    
    # Exploration coefficient: starts high, decays
    explore_weight = 0.5 * np.exp(-n_evals / 20.0)
    
    # KG-like score:
    # Base: Expected Improvement (local view)
    # Boost: Points with high sigma are more likely to reveal new minima
    #        (global view)
    
    # Normalize EI to [0, 1] roughly
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Uncertainty score: high sigma means high potential for discovery
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = sigma_safe
    
    # Combined KG-like score:
    # Weight EI by the probability of being a significant improvement,
    # and add a bonus for uncertainty in unexplored regions.
    
    # The key KG insight: value = expected reduction in the minimum
    # A point with low mu and high sigma is valuable because it might
    # be the global minimum (low mu) and we're uncertain about it (high sigma).
    
    # Score = E[improvement] * P(significant improvement) + exploration bonus
    # where P(significant improvement) ≈ Φ(z) and z = (f_best - mu)/sigma
    
    # Let's use: score = ei * (0.5 + 0.5 * p_better) + explore_weight * sigma_norm * (1 - p_better)
    # The (1 - p_better) term ensures we explore points that might be better
    # than f_best but we're not sure yet.
    
    score = ei * (0.5 + 0.5 * p_better) + explore_weight * sigma_norm * (1 - p_better)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
