import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Combines the complementarity score idea (global structure via sorted order)
    with the smooth fill-ratio based refinement from the inspirations.
    
    Args:
        demands: array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        matrix: n x n matrix where entry (i, j) is the attractiveness of 
                placing item j after item i.
    """
    start_time = time.time()
    
    n = len(demands)
    
    # Safety guard: if computation is taking too long, return a simple fallback
    # This should rarely trigger for n=120, but ensures we stay within time limits
    deadline = start_time + 290  # 290s margin before 300s limit
    
    # Vectorized calculation of pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Mask for fitting pairs
    fit_mask = pair_sums <= capacity
    
    # Fill ratio for fitting pairs
    fill_ratio = np.where(fit_mask, pair_sums / capacity, 0.0)
    
    # --- Complementarity score via sorted order ---
    # Sort demands and map each item to its ideal partner index
    sorted_demands = np.sort(demands)
    sorted_indices = np.argsort(demands)
    
    # For each item i (in original order), find its rank in sorted order
    # Then find the ideal partner: the item that best complements it to fill the bin
    # Using two-pointer approach on sorted demands
    
    # For each sorted position k, find the best partner position
    # that maximizes sum while staying <= capacity
    # This gives us a "complementarity structure"
    
    # Compute ideal partner for each sorted position
    m = len(sorted_demands)
    ideal_partner = np.zeros(m, dtype=int)
    
    # Two-pointer sweep: for each k from 0 to m-1, find the largest j
    # such that sorted_demands[k] + sorted_demands[j] <= capacity
    # We want j != k (or j can be k if it's the only option, but we prefer distinct)
    # Actually, for complementarity, we want the partner that gives the tightest fit
    
    # For each k, the ideal partner is the one that makes the sum closest to capacity
    # without exceeding it. Since sorted_demands is sorted, for each k,
    # the best partner is found by binary search or two-pointer.
    
    # Two-pointer: for k from 0 to m-1, maintain pointer j
    # j starts at m-1 and decreases as k increases (monotonic because sorted_demands is sorted)
    j_ptr = m - 1
    for k in range(m):
        # Shrink j_ptr while the pair exceeds capacity
        while j_ptr > k and sorted_demands[k] + sorted_demands[j_ptr] > capacity:
            j_ptr -= 1
        # Now j_ptr is the largest index such that sorted_demands[k] + sorted_demands[j_ptr] <= capacity
        # Handle self-pairing: we prefer a distinct partner
        if j_ptr == k:
            # Prefer a neighbor that fits
            if k + 1 < m and sorted_demands[k] + sorted_demands[k+1] <= capacity:
                ideal_partner[k] = k + 1
            elif k - 1 >= 0 and sorted_demands[k] + sorted_demands[k-1] <= capacity:
                ideal_partner[k] = k - 1
            else:
                ideal_partner[k] = k  # fallback
        else:
            ideal_partner[k] = j_ptr
    
    # Map from original index to sorted position
    # rank[i] = position of demands[i] in sorted order
    rank = np.zeros(n, dtype=int)
    for orig_idx, sorted_pos in enumerate(sorted_indices):
        rank[orig_idx] = sorted_pos
    
    # For each original pair (i, j), compute the complementarity score
    # based on how close j is to the ideal partner of i in sorted order
    
    # Create matrix of sorted positions for all pairs
    rank_i = rank[:, None]  # n x 1
    rank_j = rank[None, :]  # 1 x n
    
    # Ideal partner sorted position for each i
    ideal_partner_sorted = ideal_partner[rank_i]  # n x 1
    
    # Distance in sorted order from j's rank to i's ideal partner rank
    sorted_distance = np.abs(rank_j - ideal_partner_sorted)  # n x n
    
    # Complementarity score: high when distance is small
    # Use exponential decay: exp(-beta * distance)
    # Smaller beta = wider influence; larger beta = sharper focus on ideal partners
    beta = 0.5
    complementarity = np.exp(-beta * sorted_distance)
    
    # Only apply complementarity for fitting pairs
    complementarity = np.where(fit_mask, complementarity, 0.0)
    
    # --- Combine with fill-ratio based heuristic ---
    # Base: fill_ratio^2 (favors tight fits, smoother than ^3)
    base = fill_ratio ** 2
    
    # Smooth damping for very small pairs (waste penalty)
    # Penalize pairs with fill_ratio < 0.3 (too much waste)
    damp = 1.0 - 0.3 * np.exp(-((fill_ratio - 0.3) / 0.12) ** 2)
    
    # Combine: multiplicative blend of base and complementarity
    # When complementarity is high (j is near i's ideal partner), boost the score
    # When complementarity is low, the base score dominates
    
    # Final heuristic: base * damp * (1 + alpha * complementarity)
    # This boosts pairs that are both good fits AND structurally complementary
    alpha = 0.8  # stronger weight for complementarity enhancement
    
    heuristic = base * damp * (1.0 + alpha * complementarity)
    
    # Add a small constant to ensure positivity for fitting pairs
    heuristic = np.where(fit_mask, heuristic + 0.005, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Final safety check
    if time.time() > deadline:
        # Return a simple fallback if we're running out of time
        return np.where(pair_sums <= capacity, pair_sums / capacity, 1e-9)
    
    return heuristic
