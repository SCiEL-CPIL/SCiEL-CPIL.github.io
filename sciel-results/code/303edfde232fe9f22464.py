"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Strategy: 
    The heuristic rewards pairs (i, j) that leave a residual gap which is too 
    small to accommodate any other item in the set. This is calculated by 
    counting how many items k fit into capacity - (d_i + d_j). 
    Heuristic ~ 1 / (1 + count_of_items_that_fit_in_residual).
    """
    start_time = time.time()
    
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Deadline guard
    def deadline_check():
        return time.time() - start_time > 285

    # 1. Calculate pair sums and identify fitting pairs
    pair_sum = d[:, None] + d[None, :]
    gap = capacity - pair_sum
    fits = gap >= 0
    # Diagonal elements (i == j) are not valid pairs for "placing j after i" in the same bin 
    # as distinct items, though the ACO might theoretically pick i then i? 
    # The problem says "placing item j right after item i". Usually i != j.
    # Even if i==j is allowed by ACO logic, it's physically impossible to place the same item twice.
    # The ACO loop picks a "fitting item". If i is already in the bin, it's not "fitting" in the sense of availability.
    # However, the heuristic matrix is static. We should set diagonal to 0 or 1e-9 to discourage it.
    np.fill_diagonal(fits, False)

    # Initialize heuristic matrix
    heuristic = np.zeros((n, n), dtype=np.float64)

    # 2. For each fitting pair (i, j), count how many other items k fit into the residual gap
    # residual = capacity - d[i] - d[j]
    # We want to count k such that d[k] <= residual AND k != i AND k != j.
    
    # Vectorized approach:
    # residual_matrix[i, j] = capacity - d[i] - d[j]
    residual = gap # already computed
    
    # We need a 3D comparison: for each (i, j), count k where d[k] <= residual[i, j]
    # d is shape (n,). residual is shape (n, n).
    # We can reshape d to (1, 1, n) and residual to (n, n, 1)
    # fit_count[i, j, k] = 1 if d[k] <= residual[i, j] else 0
    
    # To save memory and time, we can do this in chunks or rely on NumPy's efficiency for n=120.
    # 120 * 120 * 120 = 1,728,000 operations. This is very small.
    
    d_col = d.reshape(1, 1, n)
    res_col = residual.reshape(n, n, 1)
    
    # Boolean matrix: True if item k fits in residual of pair (i, j)
    fits_k = d_col <= res_col
    
    # Exclude i and j themselves from the count
    # We need to mask out k == i and k == j.
    # indices:
    idx = np.arange(n)
    # k == i: for each i, j, mask out k=i.
    # We can subtract these counts after summing, or mask the boolean array.
    # Masking is safer.
    
    # Create a mask for k == i
    # For a given i, the index k should not be i.
    # This is a bit tricky to vectorize directly on the 3D array without indexing tricks.
    # Alternative: Sum all fits, then subtract the cases where k==i or k==j.
    
    total_fits = np.sum(fits_k, axis=2) # shape (n, n)
    
    # Subtract 1 if d[i] <= residual[i, j] (i fits in its own residual)
    # residual[i, j] = cap - d[i] - d[j].
    # d[i] <= cap - d[i] - d[j]  =>  2*d[i] + d[j] <= cap.
    i_fits = (2*d[:, None] + d[None, :]) <= capacity
    # Subtract 1 if d[j] <= residual[i, j] (j fits in its own residual)
    # d[j] <= cap - d[i] - d[j]  =>  d[i] + 2*d[j] <= cap.
    j_fits = (d[:, None] + 2*d[None, :]) <= capacity
    
    # Note: i_fits and j_fits are computed for all i, j.
    # We only care about pairs where fits[i, j] is True.
    
    count_other_fits = total_fits - i_fits.astype(np.float64) - j_fits.astype(np.float64)
    
    # Ensure non-negative
    count_other_fits = np.maximum(count_other_fits, 0)
    
    # 3. Calculate heuristic based on "dead gap" quality
    # Inverse proportional to count.
    # If count is 0, it's a perfect dead gap. Heuristic should be high.
    # If count is large, gap is "wasteful" (can hold more). Heuristic should be low.
    
    # Formula: 1 / (1 + count)
    # We also add a small size bias to prefer larger items, as they are harder to place.
    # Size bias: d[j] / capacity
    size_bias = d[None, :] / capacity
    
    # Combine: 
    # Base score from dead gap
    dead_gap_score = 1.0 / (1.0 + count_other_fits)
    
    # Weighted combination
    # The dead gap score is the primary driver.
    # We multiply by a size factor to break ties or slightly boost larger items.
    # Let's try: heuristic = dead_gap_score * (1 + 0.1 * size_bias)
    # Or simply additive.
    # Let's stick to the directive: "inversely proportional to the count".
    # So the core is dead_gap_score.
    
    heuristic = dead_gap_score
    
    # Apply size bias as a multiplier to encourage placing larger items (which are more constraining) 
    # into these tight spots if possible, or just generally prefer larger items.
    # Actually, if we place a large item, the residual is smaller, making it more likely to be a dead gap.
    # So the dead_gap_score naturally captures this.
    # However, if two pairs have the same count (e.g. both 0), we might prefer the one with larger d[j].
    heuristic *= (1.0 + 0.2 * size_bias)
    
    # 4. Apply constraints
    # Only valid for fitting pairs
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if deadline_check():
        return heuristic
        
    return heuristic
# EVOLVE-BLOCK-END
