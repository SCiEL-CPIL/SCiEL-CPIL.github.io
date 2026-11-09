"""Rewritten acquisition: cost-aware, two-phase, knowledge-gradient-inspired."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280 s safety margin within 300 s limit

# Precompute constants
_EPS = 1e-9

def _norm_cdf_vec(z):
    """Vectorized standard normal CDF."""
    return 0.5 * (1.0 + np.vectorize(norm.cdf)(z))

def _norm_pdf_vec(z):
    """Vectorized standard normal PDF."""
    return np.vectorize(norm.pdf)(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing cosine8 8d.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're close to the limit, return a safe default
    if time.time() > _DEADLINE - 5.0:
        # Fallback: pick the point with lowest cost and decent improvement
        scores = -cost
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = mu.shape[0]

    # Stabilize sigma
    sigma_safe = np.maximum(sigma, 1e-6)

    # Standardized improvement distance
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Knowledge Gradient proxy: expected reduction in f_best
    # When mu < f_best, the expected new best is roughly min(f_best, mu)
    # KG ≈ E[max(f_best - min(f_best, X), 0)] where X ~ N(mu, sigma^2)
    # This is similar to EI but accounts for the probability of beating f_best
    # Use a blend: KG-like term that rewards both improvement and uncertainty
    kg_proxy = ei + 0.3 * sigma_safe * norm.pdf(z)

    # Cost awareness:
    # cost is in [1, 2] (1 + mean(x_scaled_to_[0,1]))
    # We want to penalize high cost, but not so much that we always pick the cheapest
    # Use a budget-aware penalty: as we spend more, cost matters more
    # cost ranges roughly [1, 2], normalize to [0, 1]
    cost_norm = (cost - 1.0) / 1.0  # [0, 1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)

    # Two-phase weighting:
    # Early: explore (weight uncertainty more), late: exploit (weight cost more)
    # Use f_best as a proxy for progress: lower f_best = later in optimization
    # cosine8 min is -0.8, start is around 0.0 to 0.5
    # Progress proxy: how close f_best is to -0.8
    progress = np.clip((f_best - (-0.8)) / 1.0, 0.0, 1.0)  # 1.0 = early, 0.0 = late
    # Invert: early_weight is high when progress is high (early)
    early_weight = progress
    late_weight = 1.0 - progress

    # Score: combine EI/KG with cost penalty
    # Early: score = KG_proxy (explore)
    # Late: score = KG_proxy - late_weight * cost_penalty
    # cost_penalty scales with cost_norm
    cost_penalty = late_weight * 0.5 * cost_norm

    scores = kg_proxy - cost_penalty

    # Additional: heavily penalize points that are very expensive AND have low EI
    # This prevents picking a cheap-but-uninformative point over a slightly-more-expensive
    # but much-more-informative point
    # Use a multiplicative gate: if EI is very low, scale down the score
    ei_scale = np.clip(ei / (np.max(ei) + 1e-9), 0.1, 1.0)
    scores = scores * (0.5 + 0.5 * ei_scale)

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Add a tiny bit of noise to break ties and encourage diversity
    # (deterministic based on index to be reproducible)
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, size=n)
    scores = scores + noise

    return scores
# EVOLVE-BLOCK-END
