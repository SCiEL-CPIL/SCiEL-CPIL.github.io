"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix using co-occurrence frequency from FFD simulations,
    combined with gap utility scoring.
    
    Strategy:
    1. Run multiple FFD simulations to learn which item pairs frequently appear in the same bin.
    2. Use co-occurrence counts as a prior for attractiveness.
    3. Combine with gap-based scoring: favor tight packing and penalize dead gaps.
    4. Return the combined heuristic matrix.
    """
    start_time = time.time()
    time_limit = 300.0  # Keep under 300s total
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Pair sums and validity
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    remaining = capacity - pair_sums
    min_item = np.min(demands)
    
    # Initialize with small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # --- Step 1: Run FFD simulations to get co-occurrence frequencies ---
    # We'll run a batch of FFD runs with random tie-breaking to sample packings
    num_ffd_runs = 500  # Number of FFD simulations
    
    # Co-occurrence count matrix
    cooccur = np.zeros((n, n), dtype=np.int64)
    
    # Sort indices by demand descending for FFD
    sorted_idx = np.argsort(-demands)
    
    for run in range(num_ffd_runs):
        # Check time limit
        if time.time() - start_time > time_limit * 0.5:
            break
        
        # FFD: process items in descending order, place in first bin that fits
        bins = []  # List of (current_sum, [item_indices])
        placed = np.zeros(n, dtype=bool)
        
        for idx in sorted_idx:
            item_size = demands[idx]
            # Find first bin that can fit this item
            placed_in_bin = False
            for b_idx, (bin_sum, bin_items) in enumerate(bins):
                if bin_sum + item_size <= capacity:
                    # Place in this bin
                    bins[b_idx] = (bin_sum + item_size, bin_items + [idx])
                    # Count co-occurrences with all items already in this bin
                    for other_idx in bin_items:
                        cooccur[idx, other_idx] += 1
                        cooccur[other_idx, idx] += 1
                    placed_in_bin = True
                    break
            
            if not placed_in_bin:
                # Open new bin
                bins.append((item_size, [idx]))
        
        # After FFD completes, we've counted co-occurrences during placement
        # (each item is counted with items already in the bin when placed)
    
    # Normalize co-occurrence counts to [0, 1] range
    max_cooccur = np.max(cooccur)
    if max_cooccur > 0:
        cooccur_norm = cooccur / max_cooccur
    else:
        cooccur_norm = np.zeros_like(cooccur)
    
    # --- Step 2: Gap-based scoring ---
    # Perfect fits
    perfect_fit = valid & (remaining < 1e-9)
    # Usable gaps (can fit at least one more item)
    usable_gap = valid & (remaining >= min_item)
    # Dead gaps (small positive but can't fit any item)
    dead_gap = valid & (remaining > 1e-9) & (remaining < min_item)
    
    # Score for usable gaps: inverse of remaining space
    score_usable = 1.0 / (remaining[usable_gap] + 1.0)
    
    # Score for perfect fits: strong boost
    score_perfect = 10.0
    
    # Score for dead gaps: very low
    score_dead = 1e-9
    
    # Apply gap scores
    gap_eta = np.full((n, n), 1e-9, dtype=float)
    gap_eta[usable_gap] = score_usable
    gap_eta[perfect_fit] = score_perfect
    gap_eta[dead_gap] = score_dead
    
    # --- Step 3: Combine co-occurrence and gap scores ---
    # Weighted combination: primarily use co-occurrence, with gap as tiebreaker
    # Only for valid pairs
    eta[valid] = 0.7 * cooccur_norm[valid] + 0.3 * gap_eta[valid]
    
    # Ensure invalid pairs stay at 1e-9
    eta[~valid] = 1e-9
    
    # Diagonal: self-pairing is not meaningful
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
