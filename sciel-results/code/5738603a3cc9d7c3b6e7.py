"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce deadline
_start_time = time.time()
_DEADLINE = 150.0  # Seconds
_MARGIN = 1.0      # Safety margin

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # Fallback: return best fit quickly or just return zeros/invalid
        # To be safe, return -inf for all to trigger fallback in evaluator if it exists,
        # or just return the best fit calculation which is fast.
        # Since the calculation is O(N), it's fast. The guard is just a precaution.
        pass

    # Filter bins with enough remaining capacity
    remaining = bins - item
    mask = remaining >= 0
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e10, dtype=float)
    
    if np.any(mask):
        # Hybrid Best Fit / Worst Fit
        # Primary: Minimize remaining space (Best Fit)
        # Secondary: Maximize bin size (Worst Fit component) to preserve small bins
        # Score = -remaining + alpha * bin_size
        # Since remaining = bin_size - item
        # Score = -(bin_size - item) + alpha * bin_size
        # Score = item + (alpha - 1) * bin_size
        # For alpha < 1, we still prefer smaller bin_size (Best Fit dominant).
        # For alpha = 0.1, Score = item - 0.9 * bin_size.
        # This strongly prefers smaller bins, but less strongly than pure best fit?
        # Wait.
        # Pure Best Fit: Score = -remaining = -(bin_size - item) = item - bin_size.
        # Hybrid: Score = item - 0.9 * bin_size.
        # Difference: Hybrid is less negative for larger bins.
        # So it prefers larger bins *slightly* more than pure best fit, among bins that fit.
        # Actually, let's look at the gradient.
        # Pure: d(score)/d(bin_size) = -1.
        # Hybrid: d(score)/d(bin_size) = -0.9.
        # So the penalty for a larger bin is slightly reduced.
        
        alpha = 0.1
        valid_bins = bins[mask]
        valid_remaining = remaining[mask]
        
        # Score = -valid_remaining + alpha * valid_bins
        scores[mask] = -valid_remaining + alpha * valid_bins
        
        # Add small tie-breaker for determinism
        scores[mask] += np.arange(len(bins))[mask] * 1e-9

    return scores
# EVOLVE-BLOCK-END
