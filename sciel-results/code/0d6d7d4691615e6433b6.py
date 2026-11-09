"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Best-fit with a smooth penalty for mid-range remaining capacity.
    - Primary signal: minimize remaining capacity after placement (best-fit).
    - Secondary signal: penalize remaining capacity in a "wasteful" mid-range
      band using a smooth (log-sum-exp style) surrogate so the objective is
      differentiable and avoids hard kinks.
    - Tertiary signal: small bonus for bins already partially filled to
      encourage consolidating into existing bins.
    """
    remaining = bins - item
    valid = remaining >= 0

    # Base score: negative remaining capacity (best-fit) — tighter is better.
    score = -remaining.copy()

    # Invalidate bins that cannot hold the item.
    score[~valid] = -np.inf

    if not np.any(valid):
        return score

    post = remaining[valid]  # remaining capacity after placing item

    # Smooth penalty for mid-range remaining capacity.
    # We want to discourage leaving a "moderate" amount of space (e.g., 10–50)
    # because that space is unlikely to be fully utilized later.
    # Use a smooth bump centered at c=30 with width w=15:
    #   penalty = w * log(1 + exp(-(post - c)/w)) - w*log(2)
    # This is a smooth approximation to -w * min(0, c - post) + const,
    # but we actually want a penalty that peaks in the mid-range and tapers
    # off for both very small and very large remaining.
    # A simpler smooth approach: penalize the squared distance from an
    # ideal "near-zero" remaining, but only for post > small threshold.
    # We use: penalty = alpha * softplus(post - tau) where tau is a small
    # threshold and softplus(x) = log(1 + exp(x)) / 1 (scaled).
    # This gives near-zero penalty for post <= tau and grows smoothly for
    # post > tau, encouraging tight fits without a hard kink.
    tau = 2.0  # threshold: remaining below this is "good enough"
    alpha = 0.3  # strength of the smooth penalty
    x = (post - tau) / 1.0  # scale for numerical stability
    # softplus(x) = log(1 + exp(x)), computed stably
    softplus = np.where(x > 20, x, np.log1p(np.exp(x)))
    penalty = alpha * softplus

    # Small bonus for bins already partially filled (encourage consolidation).
    # utilization_before = (bins - post) / bins = item / bins
    # Use a gentle bonus that saturates to avoid over-weighting tiny bins.
    bins_valid = bins[valid]
    util = np.where(bins_valid > 1e-9, item / bins_valid, 0.0)
    # Smooth saturation: bonus = beta * (1 - exp(-gamma * util))
    beta = 0.15
    gamma = 2.0
    bonus = beta * (1.0 - np.exp(-gamma * util))

    score[valid] -= penalty
    score[valid] += bonus

    return score
# EVOLVE-BLOCK-END
