"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adaptive fragmentation-aware bin selection.

    Core idea:
      - Primary signal: best-fit (minimize leftover after placing the item).
      - Adaptive penalty: discourage bins whose post-placement leftover is
        "wasted" (too small to be useful for future Weibull-distributed items)
        while also avoiding over-fragmentation.

    The penalty is calibrated from the current state of all bins:
      - Compute the distribution of current fill levels (bins values).
      - Estimate the "typical" future item size from the Weibull shape.
      - Penalize gaps that are in a "dead zone" (small but not zero).
    """
    start_time = time.time()
    deadline = 115.0  # 120s limit, 5s safety margin

    capacity = 100.0

    # Compute remaining capacity if we place the item in each bin
    remaining = bins - item

    # Valid bins are those with remaining >= 0
    valid_mask = remaining >= 0

    # If no valid bin, return -inf for all
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)

    # Initialize scores
    scores = np.full_like(bins, -np.inf)

    # Get the current fill levels of valid bins
    valid_bins = bins[valid_mask]
    valid_remaining = remaining[valid_mask]

    # Estimate a "useful gap" threshold from the data.
    # Weibull items with typical shape k~1.5-2.5, scale ~10-20 have
    # mean ~10-15. A gap smaller than ~half the expected next item is
    # likely wasted. We use a data-driven estimate:
    #   - Look at the spread of current bin fill levels
    #   - The median fill level gives a sense of typical item sizes
    #
    # For Weibull(1000 items, capacity 100), typical item sizes are ~8-15.
    # Use a smoothed estimate: the 25th percentile of valid bin fill levels
    # as a proxy for "small item" size.
    if len(valid_bins) > 0:
        # Use a robust estimate of typical small-item size
        # The lower quantile of fill levels approximates smaller items
        q25 = np.percentile(valid_bins, 25)
        # Clamp to a reasonable range [3, 15]
        small_item_est = np.clip(q25, 3.0, 15.0)
    else:
        small_item_est = 10.0

    # "Dead zone" threshold: gaps smaller than this are likely wasted
    dead_zone = 0.5 * small_item_est

    # Compute adaptive fragmentation penalty:
    # For each valid bin, after placing the item, the gap is valid_remaining.
    # If the gap is in the dead zone (0 < gap < dead_zone), apply a penalty.
    # If the gap is very small (< 0.5), it's fine (tight fit).
    # If the gap is large, no penalty (flexibility for future items).
    #
    # Use a smooth penalty: penalty = -w * exp(-((gap - dead_zone/2)^2) / (2*sigma^2))
    # This creates a Gaussian-shaped penalty centered in the dead zone.
    sigma = 0.3 * small_item_est
    gap_center = dead_zone * 0.5  # center of the dead zone
    penalty_width = sigma

    # Smooth Gaussian penalty (negative, so it reduces score)
    gap_deviation = valid_remaining - gap_center
    gaussian_penalty = np.exp(-0.5 * (gap_deviation / penalty_width) ** 2)

    # Scale the penalty: stronger when we have many bins (more fragmentation risk)
    # and weaker when few bins are used
    n_valid = len(valid_bins)
    n_total = len(bins)
    # Penalty weight: increases with number of open bins relative to total
    # Use a soft saturation
    frag_weight = 1.5 * (1.0 - np.exp(-0.1 * n_valid)) if n_total > 0 else 0.5

    # Base score: best-fit (negative of remaining, so tighter = higher score)
    base_score = -1.0 * valid_remaining

    # Add a mild consolidation bonus for fuller bins
    # This helps when multiple bins have similar remaining space
    consolidation = 0.03 * valid_bins

    # Combine: base + consolidation - fragmentation_penalty
    frag_penalty = frag_weight * gaussian_penalty

    scores_valid = base_score + consolidation - frag_penalty

    # Assign scores
    scores[valid_mask] = scores_valid

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores

# EVOLVE-BLOCK-END
