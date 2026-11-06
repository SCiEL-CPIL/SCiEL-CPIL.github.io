"""Online bin packing – priority scoring for FunSearch/EoH contract."""
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin that has enough remaining capacity for `item`.
    The item is placed in the bin with the highest score.

    Strategy: Best-Fit Decreasing style with a smooth penalty.
    We prefer bins where the item fits snugly (leaves minimal slack),
    but we also avoid filling bins too much if that would leave small
    unusable gaps. We use a smooth surrogate: for each bin, compute the
    remaining capacity after placing the item (slack). Lower slack is
    generally better (Best Fit), but extremely small slack relative to
    typical item sizes can be bad if it creates orphan fragments.

    For Weibull-distributed items with capacity 100, items are typically
    moderate. A simple Best-Fit (minimize slack) works well. To improve,
    we can add a small penalty for bins that are already very full, to
    encourage spreading items and keeping more bins partially available.

    Score = -slack + epsilon * (bins / capacity)
    where slack = bins - item, and bins/capacity is how full the bin already is.
    This slightly prefers less-full bins among those with similar slack.
    """
    remaining = bins - item
    mask = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    fill = bins[mask]  # current fill level
    
    # Best-Fit: minimize slack (maximize fill after placement)
    # Slack = bins - item, so -slack = item - bins
    # We want to maximize (item - bins) = -slack, i.e., minimize slack.
    # But also add a tiny bonus for less-full bins to avoid fragmentation.
    # Score = -slack + small_penalty * fill
    # Since slack = bins - item, -slack = item - bins
    # So score = item - bins + alpha * bins = item + (alpha - 1) * bins
    # To prefer less-full bins when slack is similar, we want score to
    # decrease as bins increases, so (alpha - 1) should be negative, i.e., alpha < 1.
    # Let alpha = 0.9, so score = item + (0.9 - 1) * bins = item - 0.1 * bins
    # This means among bins with the same slack, less-full bins score higher.
    
    # Use a nonlinear penalty that increases as bins get fuller
    # This better captures the trade-off between best-fit and keeping bins available
    # The penalty grows quadratically with fill level, making it stronger for nearly-full bins
    fill_ratio = fill / 100.0  # Normalize to [0, 1] assuming capacity 100
    penalty = 0.3 * fill_ratio**2  # Quadratic penalty for fullness
    scores[mask] = item - penalty
    
    return scores
