"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Adaptive Strategy:
    # Estimate the typical item size from the current state of the bins.
    # The 'filled' amounts represent the sizes of items already placed.
    filled = 100.0 - bins
    med = np.median(filled)
    
    # If the current item is smaller than the median, it is likely to fit into
    # existing gaps. Use Best Fit to minimize wasted space in small gaps.
    # If the current item is larger than the median, it is a "large" item.
    # Placing it in a small gap is impossible (handled by mask), but placing it
    # in a medium gap might fragment it. Placing it in a large gap (Worst Fit)
    # preserves the integrity of the bin for potential future large items or
    # allows the bin to be filled more completely by this large item.
    # Actually, for large items, we want to put them in bins that are already
    # somewhat full to avoid creating new half-full bins?
    # No, for large items, Best Fit is usually fine too.
    # But Worst Fit for large items ensures that if we *do* open a new bin,
    # it's because all existing bins are too small, and we keep the "large"
    # bins available?
    
    # Let's refine:
    # If item is small: Best Fit (minimize gap).
    # If item is large: Best Fit is still generally good for packing density.
    # However, the "Stochastic Inventory" hint suggests valuing the remainder.
    
    # Let's try the "Mode Matching" approach which is more distinct.
    # Calculate a histogram of filled items to find the mode.
    # For speed, use bins of the filled amounts.
    
    # But the Adaptive Threshold is a classic structural change.
    # Let's implement:
    # Small items -> Best Fit
    # Large items -> Worst Fit (or a variant)
    
    # Why Worst Fit for large?
    # If we put a large item in the best fit bin, we create a small gap.
    # If we put it in the worst fit bin, we create a large gap.
    # For large items, the probability of a future item fitting in a small gap is low.
    # The probability of a future item fitting in a large gap is higher.
    # So keeping a large gap alive is better for large items?
    # This is the logic of Worst Fit Decreasing.
    
    if item < med:
        # Best Fit
        return -(bins - item)
    else:
        # Worst Fit
        return bins - item

# EVOLVE-BLOCK-END
