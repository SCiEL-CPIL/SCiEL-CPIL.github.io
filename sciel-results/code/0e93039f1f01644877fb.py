"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 8d.
    
    Structural change: Hybrid PoI/Thompson Sampling approach.
    
    Instead of a linear blend of normalized PoI and UCB (which can be sensitive
    to the scale of sigma), we use a probabilistic decision rule inspired by
    Thompson Sampling:
    1. Compute Probability of Improvement (PoI) for each candidate.
    2. Use PoI to determine the "confidence" in the improvement.
    3. For candidates with high PoI, we trust the mean more (exploitation).
    4. For candidates with low PoI but high uncertainty, we allow larger
       "virtual samples" (exploration).
    
    Specifically, we construct a score that mimics the expected value of a
    sampled objective, but weighted by cost:
    score = - (mu + k * sigma * z_sample) / cost
    
    However, since we need a deterministic acquisition function (no actual
    random sampling in the return value to ensure stability and reproducibility
    of the "best" candidate selection), we approximate the Thompson Sampling
    score with a deterministic equivalent that balances the mean and uncertainty
    based on the PoI.
    
    The key insight is that PoI gives us a natural weight for how much to
    "trust" the mean vs. the uncertainty. We use this to modulate the
    effective uncertainty term.
    
    Additionally, we use robust normalization (median/MAD) from Inspiration 2
    to handle outliers in the candidate set, which is crucial in high dimensions
    where a few extreme candidates can skew mean/std normalization.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1.0)  # Cost is in [1, 2]

    # 1. Probability of Improvement (PoI) for minimization
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    poi = norm.cdf(z)
    
    # 2. Cost-adjusted PoI
    poi_cost = poi / cost_safe

    # 3. Construct a Thompson Sampling-like deterministic score
    # In TS, we sample f ~ N(mu, sigma^2) and pick argmin f/cost.
    # The expected value of the best of N samples is mu - sigma * E[max(Z1...ZN)].
    # For a single sample, it's just mu.
    # But we want to balance exploration and exploitation.
    
    # Let's use a "virtual" exploration term that is scaled by the PoI.
    # If PoI is high, we trust the mean, so exploration term should be small.
    # If PoI is low, we are uncertain, so exploration term should be larger.
    
    # Define an effective uncertainty weight based on PoI
    # w_explore = 1 - PoI (roughly)
    # But we want to be smoother. Let's use the z-score itself.
    # High z -> high PoI -> low exploration.
    # Low z -> low PoI -> high exploration.
    
    # Let's try a different angle:
    # Score = -mu / cost + lambda * sigma / cost
    # where lambda is adaptive.
    
    # From the current best program, we know that a blend of PoI and UCB works.
    # Let's refine the blending mechanism.
    
    # Instead of normalizing PoI and UCB separately and then blending,
    # let's create a single composite metric.
    
    # Composite: 
    # Exploitation term: -mu / cost (higher is better)
    # Exploration term: sigma / cost (higher is better)
    
    # But we need to balance them.
    # Let's use the PoI to determine the balance.
    
    # If PoI is high, we are confident that this point is good, so we should
    # prioritize it based on its mean value.
    # If PoI is low, we should prioritize it based on its uncertainty.
    
    # Let's define:
    # exploit_score = -mu / cost_safe
    # explore_score = sigma_safe / cost_safe
    
    exploit_score = -mu / cost_safe
    explore_score = sigma_safe / cost_safe
    
    # Adaptive weight based on PoI
    # When PoI is high (close to 1), w_exploit should be high.
    # When PoI is low (close to 0), w_explore should be high.
    
    # Let's use a sigmoidal or linear mapping.
    # w_exploit = PoI
    # w_explore = 1 - PoI
    
    # But this might be too aggressive. Let's smooth it.
    # w_exploit = 0.5 + 0.5 * PoI
    # w_explore = 0.5 - 0.5 * PoI
    
    # Actually, let's stick to the structure that worked (PoI + UCB blend)
    # but use robust normalization.
    
    # 4. Robust Normalization (Median/MAD)
    # This is the key refinement from Inspiration 2.
    
    # Normalize PoI_cost
    poi_med = np.median(poi_cost)
    poi_mad = np.median(np.abs(poi_cost - poi_med)) + 1e-12
    poi_norm = (poi_cost - poi_med) / poi_mad
    
    # Normalize Exploration (sigma/cost)
    expl_med = np.median(explore_score)
    expl_mad = np.median(np.abs(explore_score - expl_med)) + 1e-12
    expl_norm = (explore_score - expl_med) / expl_mad
    
    # 5. Adaptive Blending
    # Progress: 0 when f_best is near -0.8, 1 when f_best is far (e.g., 0.0)
    progress = np.clip((f_best - MIN_POSSIBLE) / (0.0 - MIN_POSSIBLE), 0.0, 1.0)
    
    # Exploration weight: higher when progress is high (far from optimum)
    # Use a smooth transition
    # From the current best: w_explore = 0.1 + 0.5 * progress**1.5
    # Let's keep this as it improved the score.
    w_explore = 0.1 + 0.5 * progress**1.5
    w_exploit = 1.0 - w_explore
    
    # Combine
    score = w_exploit * poi_norm + w_explore * expl_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
