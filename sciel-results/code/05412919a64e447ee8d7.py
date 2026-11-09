"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: Combines best-fit (minimize remaining waste) with a smooth
    nonlinear transform and a 'Next-Item Fit' proxy that rewards bins whose
    remaining capacity is close to the median item size, reducing fragmentation.
    """
    # Feasibility: only bins with enough remaining capacity
    remaining = bins - item
    feasible = remaining >= 0
    
    # If no feasible bins, return all -inf
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Primary objective: minimize waste (best-fit)
    # Use a smooth nonlinear transform: -remaining^1.5
    # This strengthens preference for tight fits compared to linear -remaining
    # while remaining smooth (no cliffs)
    
    # Secondary objective: 'Next-Item Fit' proxy
    # Estimate median item size from the current bin states
    # The median of used capacities (bins - remaining) is a proxy for typical item size
    # We want remaining capacity to be close to this typical size to accommodate future items
    used = np.where(feasible, bins - remaining, 0.0)
    
    # Use the median of used capacities as a proxy for typical item size
    # If all bins are empty or nearly so, fall back to a small value
    median_item_size = np.median(used[feasible]) if np.any(feasible) else 1.0
    # Ensure median_item_size is positive
    median_item_size = max(median_item_size, 0.1)
    
    # Score for next-item fit: negative squared difference between remaining and median_item_size
    # Bins with remaining close to median_item_size get higher scores
    # This encourages keeping bins in a state where they can accept typical future items
    next_item_fit = -(remaining - median_item_size) ** 2
    
    # Combine: primary (best-fit) + secondary (next-item fit)
    # Weight the best-fit more heavily since it's the primary objective
    # The next-item fit is a smaller correction to avoid fragmentation
    score = np.where(feasible, -remaining ** 1.5 + 0.1 * next_item_fit, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
