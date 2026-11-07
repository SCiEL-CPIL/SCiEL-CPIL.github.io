"""Rewritten program: Inverse-waste heuristic for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    
    The heuristic favors pairs of items (i, j) that fit in the same bin and
    have minimal waste (i.e., their combined size is as close to the capacity
    as possible). This encourages the ants to pack bins tightly, which is a
    strong strategy for minimizing the total number of bins.
    
    Args:
        demands: Array of item sizes (floats).
        capacity: Bin capacity (float).
        
    Returns:
        A 2D numpy array of shape (n, n) where entry (i, j) is the
        attractiveness of placing item j after item i.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate the sum of sizes for every pair (i, j)
    pair_sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit within the capacity
    fits = pair_sums <= capacity
    
    # Calculate the waste for each pair (capacity - sum)
    # For pairs that don't fit, waste is negative, but we will mask them out.
    waste = capacity - pair_sums
    
    # Initialize the heuristic matrix with a small positive value (1e-9)
    # This handles the "not positive" requirement for non-fitting or invalid entries.
    heuristic = np.full((n, n), 1e-9, dtype=float)
    
    # For pairs that fit, calculate the attractiveness.
    # We use 1 / (waste + 1) to strongly favor pairs with zero or low waste.
    # This avoids division by zero and creates a smooth decay as waste increases.
    # We use np.where to ensure we only update the entries that fit.
    # To avoid calculating 1/(negative + 1) which could be negative or >1,
    # we compute on a masked array or use where carefully.
    
    # Safe computation:
    # Create a temporary array with waste for fitting pairs and a large value for others
    # to prevent unwanted updates, or just compute and mask.
    # Since waste is negative for non-fitting, 1/(waste+1) could be negative or >1.
    # So we must strictly mask.
    
    # Calculate values for all, but we will only take the fitting ones.
    # To avoid warnings/errors with division, we add 1 to waste.
    # If waste is -100 (non-fit), 1/(-99) is negative.
    # We will discard these via the mask.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_heuristic = 1.0 / (waste + 1.0)
        
    # Apply the mask: keep raw_heuristic where fits is True, else 1e-9
    heuristic = np.where(fits, raw_heuristic, 1e-9)
    
    # Ensure no negative values leak through (shouldn't happen with fits mask, 
    # but good for safety)
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
