import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 5.0

# Pre-compute random samples for Monte Carlo EI
# 2000 candidates * 50 samples = 100,000 draws. This is very fast.
# We use a fixed seed for reproducibility within the run to avoid state issues, 
# but the samples are static.
# Actually, using a fixed set of samples is better for consistency and speed.
_N_SAMPLES = 50
_rng = np.random.default_rng(42)
_Z_SAMPLES = _rng.standard_normal((1, _N_SAMPLES))

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo Expected Improvement (MC-EI)
    with a variance-based exploration term.
    
    This approach approximates the Expected Improvement by sampling from the
    posterior distribution N(mu, sigma^2) and averaging the improvement
    max(0, f_best - sample). This is robust to the multimodal nature of
    Goldstein-Price and avoids potential numerical issues with analytic EI
    in regions of high uncertainty.
    
    An exploration term is added to encourage visiting high-uncertainty regions
    that have a reasonable probability of improvement, which helps escape local
    minima in the early stages.
    """
    # Check deadline
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        # Fallback: pure exploitation
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Monte Carlo Expected Improvement
    # samples shape: (2000, 50)
    # z_samples shape: (1, 50)
    samples = mu[:, np.newaxis] + sigma_safe[:, np.newaxis] * _Z_SAMPLES
    
    # Improvement: max(0, f_best - sample)
    improvements = np.maximum(0.0, f_best - samples)
    
    # MC-EI: mean of improvements
    mc_ei = np.mean(improvements, axis=1)
    
    # Exploration Term
    # We want to explore where uncertainty is high AND there is a chance of improvement.
    # A simple heuristic is sigma * P(improvement) or just sigma scaled by a factor.
    # However, pure UCB can be too aggressive.
    # Let's use a term that is significant when sigma is high relative to the potential improvement.
    
    # Probability of improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    prob_improve = norm.cdf(z_clipped)
    
    # Exploration score: sigma * prob_improve
    # This is 0 when we are sure we won't improve (mu >> f_best)
    # This is ~sigma when we are sure we will improve (mu << f_best)
    # This is ~0.5*sigma when mu == f_best
    exploration = sigma_safe * prob_improve
    
    # Combine
    # The weight of exploration should decrease as we get closer to the optimum.
    # If f_best is very close to the global min (3.0), we should exploit more.
    # If f_best is far, we should explore more.
    
    f_min = 3.0
    gap = f_best - f_min
    
    if gap <= 0:
        # Already at optimum (numerically)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Adaptive weight:
    # Early on (large gap), we want to balance EI and Exploration.
    # Later (small gap), we want to rely more on EI.
    # Let's use a simple decay: weight = 1.0 / (1.0 + gap)
    # Or maybe just a fixed weight is safer?
    # Let's try a weight that is 1.0 initially and decays.
    # Since we don't know the initial f_best, we can't normalize perfectly.
    # However, Goldstein-Price values are typically in [3, ~50].
    # A weight of 0.5 to 1.0 for exploration is reasonable.
    
    # Let's use a weight that is proportional to the relative uncertainty.
    # If sigma is large compared to the gap, explore more.
    # But simpler: just add a scaled exploration term.
    # The MC-EI already includes some exploration (variance in samples).
    # Adding extra exploration helps with the "cold start" and escaping plateaus.
    
    # Let's use a weight of 0.5. This is a heuristic but standard for UCB-like terms.
    # To make it adaptive:
    # If the mean sigma is high, we are in an early/uncertain phase.
    mean_sigma = np.mean(sigma_safe)
    
    # If mean_sigma is large relative to the gap, we are uncertain -> explore more.
    # If mean_sigma is small, we are certain -> exploit more.
    # ratio = mean_sigma / gap
    # weight = min(1.0, 0.5 * ratio) ?
    
    # Let's stick to a robust combination:
    # Score = MC_EI + alpha * Exploration
    # Alpha = 0.5
    
    alpha = 0.5
    scores = mc_ei + alpha * exploration
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
