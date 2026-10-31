"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

CAPACITY = 100.0
# Weibull parameters - using values that seem reasonable for the task
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 25.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Treat the bin remainders as a continuous probability density function (PDF)
    and score each candidate bin by the negative value of this empirical PDF at the
    resulting remainder, thereby prioritizing placements that increase the local density
    of gaps in the "hot zone" where future items are most likely to fit.
    
    This approach leverages the statistical mode of the existing gap distribution to
    encourage consolidation into bins that will create remainders matching the expected
    item size distribution, reducing wasted capacity.
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # Estimate the empirical PDF of the current bin remainders
    # We use a simple kernel density estimate with a Gaussian kernel
    # For efficiency, we use a histogram-based approximation
    
    # Get all positive remainders to estimate the distribution
    all_remainders = bins[bins > 0]
    
    if len(all_remainders) < 2:
        # Not enough data for PDF estimation, fall back to best fit
        scores[feasible] = -feasible_gaps
        return scores
    
    # Create a grid for PDF estimation
    # Use a reasonable range from 0 to max(CAPACITY, max remainder)
    max_val = max(CAPACITY, np.max(all_remainders))
    grid_size = 50
    grid = np.linspace(0, max_val, grid_size)
    grid_width = grid[1] - grid[0]
    
    # Compute histogram as PDF approximation
    hist_counts, _ = np.histogram(all_remainders, bins=grid)
    pdf = hist_counts / (len(all_remainders) * grid_width)
    
    # For each feasible gap, compute the PDF value at that gap
    # We interpolate from the grid
    # To avoid issues at boundaries, clip values
    feasible_gaps_clipped = np.clip(feasible_gaps, 0, max_val)
    
    # Interpolate PDF values
    pdf_values = np.interp(feasible_gaps_clipped, grid, pdf)
    
    # Score is negative PDF value (we want to maximize PDF, so minimize -PDF)
    # But we also need to consider that very small gaps might be problematic
    # Add a small penalty for gaps that are too small to be useful
    
    # The "useful" gap range is roughly where future items can fit
    # For Weibull with k=2, lambda=25, the mode is at lambda*(k-1)^(1/k) = 25*(1)^(0.5) = 25
    # So gaps around 25 are most valuable
    
    # Penalize gaps that are much smaller than the expected item size
    # This encourages consolidation
    penalty_small_gap = np.where(feasible_gaps < _WEIBULL_LAMBDA * 0.3, 
                                  (_WEIBULL_LAMBDA * 0.3 - feasible_gaps) * 0.1, 
                                  0)
    
    # Final score: negative PDF value plus small gap penalty
    final_scores = -pdf_values + penalty_small_gap
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
