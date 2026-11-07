"""Cost-aware EI acquisition with adaptive cost penalty for three-hump camel 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Higher score -> evaluate next.
    # Adaptive cost penalty: estimate GP predictive variance at origin (known ~0)
    # to balance exploration vs. expensive evaluations.

    n = mu.size

    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)

    # Standard EI in objective units
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Estimate GP predictive variance at the origin (x=0, known f≈0).
    # At the origin, the GP posterior mean is pulled toward 0 (the known value),
    # so the residual (mu_orig - 0) reflects the GP's uncertainty there.
    # We use this as an adaptive cost penalty scale.
    #
    # The origin in [0,1] coordinates is at 0.5 in each dimension.
    # The GP mean at the origin: mu[origin]. The true value is ~0.
    # If the GP hasn't seen the origin, sigma[origin] is large and
    # mu[origin] is near the prior mean (standardized ~0 in objective units).
    #
    # We use the magnitude of the GP's mean at the origin as a proxy for
    # how "surprised" the GP is there, which correlates with uncertainty.
    # A simpler and more robust approach: use the median sigma as a scale.
    #
    # Adaptive penalty: penalize high-cost points more when the GP is
    # already confident (low sigma), and less when uncertain (high sigma).
    # This encourages exploration in uncertain, high-cost regions early
    # and exploitation in low-cost, confident regions later.
    sigma_median = np.median(sigma)
    sigma_orig = sigma_median  # proxy for uncertainty scale

    # Adaptive cost multiplier:
    # - When f_best is far from 0 (early), we want to explore more
    #   (less penalty for cost).
    # - When f_best is close to 0 (late), we want to exploit
    #   (more penalty for cost).
    # - When sigma is high (uncertain), reduce cost penalty.
    # - When sigma is low (confident), increase cost penalty.
    #
    # penalty = cost * (1 + alpha * f_best / (sigma_orig + eps))
    # But f_best can be negative (three-hump camel min is 0, but values can be
    # slightly negative due to GP noise). Clip to non-negative.
    f_best_pos = max(f_best, 0.0)

    # Use a smooth, bounded penalty that adapts to uncertainty
    # Scale factor: when sigma_orig is large relative to f_best, the GP is
    # uncertain and we should explore (lower penalty).
    # ratio = f_best_pos / (sigma_orig + 1e-9)
    # penalty_scale = 1.0 + 0.5 * ratio  (mildly increasing penalty with confidence)

    # Simpler: use the median sigma to normalize the cost penalty
    # so that the penalty adapts to the current level of GP uncertainty.
    # Early on, sigma is large -> cost penalty is relatively small -> explore.
    # Late on, sigma is small -> cost penalty is relatively large -> exploit.
    cost_norm = cost / (sigma_orig + 1e-9)
    cost_norm = np.clip(cost_norm, 0.5, 5.0)  # bound to avoid extremes

    # Final score: EI divided by adaptive cost
    score = ei / cost_norm

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
