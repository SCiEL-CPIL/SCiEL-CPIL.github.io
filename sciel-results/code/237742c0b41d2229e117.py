"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'Simulated Bin-Packing Frequency Prior'.
    
    Strategy:
    1. Run a fast, high-quality bin packing heuristic (First-Fit-Decreasing, FFD)
       to generate a reference set of bins.
    2. Analyze the reference bins to count how often item j appears in the same
       bin as item i, specifically focusing on the 'last item placed' relationship 
       or simple co-occurrence within the same bin.
    3. The heuristic eta[i, j] is set to the frequency of j appearing in a bin 
       where i was also present, normalized. 
    4. To better guide the 'next item' choice, we weight pairs where i is the 
       largest item in the bin (anchor) or simply use co-occurrence counts.
       
    This approach shifts from local geometric heuristics (gap size) to a 
    global, data-driven prior derived from a known-good packing structure.
    """
    start_time = time.time()
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Safety check for time
    if time.time() - start_time > 280:
        return np.full((n, n), 1e-9)
    
    # 1. Generate a reference packing using First-Fit-Decreasing (FFD)
    # FFD is known to produce near-optimal results for many instances.
    # We simulate the process to extract pair statistics.
    
    # Sort indices by size descending for FFD
    sorted_indices = np.argsort(-demands)
    
    # Bins: list of lists of item indices
    bins = []
    bin_sums = []
    item_to_bin = {}
    
    for idx in sorted_indices:
        size = demands[idx]
        if size > capacity:
            # Item too big, skip or handle separately (task says 20-99, cap 150, so all fit)
            continue
            
        placed = False
        for b_idx in range(len(bins)):
            if bin_sums[b_idx] + size <= capacity:
                bins[b_idx].append(idx)
                bin_sums[b_idx] += size
                item_to_bin[idx] = b_idx
                placed = True
                break
        
        if not placed:
            new_bin_idx = len(bins)
            bins.append([idx])
            bin_sums.append(size)
            item_to_bin[idx] = new_bin_idx
    
    # 2. Analyze pairs
    # We want eta[i, j] to be high if j is a good follow-up to i.
    # In the FFD simulation, items are added in sorted order.
    # However, the ACO agent places items in a different order.
    # A robust static heuristic is the 'co-occurrence' frequency:
    # How often do i and j end up in the same bin?
    # Even better: How often is j in the same bin as i, given i is 'large'?
    
    # Let's compute a co-occurrence matrix.
    # co_occ[i, j] = 1 if i and j are in the same bin, else 0.
    # To make it a frequency prior, we can count how many times this specific
    # combination appears across multiple diverse heuristic runs (e.g., FFD, Best-Fit, Random-Fit)
    # to get a more robust signal.
    
    # Let's run 3 different heuristics to get a consensus prior.
    # Heuristic 1: FFD (already done above)
    # Heuristic 2: Best-Fit Decreasing (BFD)
    # Heuristic 3: First-Fit (FF) without sorting (Random order simulation)
    
    co_occurrence_counts = np.zeros((n, n), dtype=float)
    
    # Helper to extract co-occurrence from a list of bins
    def extract_co_occurrence(bins_list):
        counts = np.zeros((n, n), dtype=float)
        for b in bins_list:
            if len(b) > 1:
                # All pairs in this bin co-occur
                for i in b:
                    for j in b:
                        if i != j:
                            counts[i, j] += 1
        return counts
    
    # Add FFD counts
    co_occurrence_counts += extract_co_occurrence(bins)
    
    # Run BFD
    bins_bfd = []
    bin_sums_bfd = []
    for idx in sorted_indices:
        size = demands[idx]
        best_fit_bin = -1
        best_fit_remaining = float('inf')
        
        for b_idx in range(len(bins_bfd)):
            rem = capacity - bin_sums_bfd[b_idx]
            if size <= rem and rem < best_fit_remaining:
                best_fit_bin = b_idx
                best_fit_remaining = rem
        
        if best_fit_bin != -1:
            bins_bfd[best_fit_bin].append(idx)
            bin_sums_bfd[best_fit_bin] += size
        else:
            bins_bfd.append([idx])
            bin_sums_bfd.append(size)
            
    co_occurrence_counts += extract_co_occurrence(bins_bfd)
    
    # Run FF with random order (seeded for reproducibility in heuristic calc, though 
    # the ACO is seeded per instance, the heuristic is static. We just need a good prior.)
    rng = np.random.RandomState(42)
    random_order = rng.permutation(n)
    
    bins_ff_rand = []
    bin_sums_ff_rand = []
    for idx in random_order:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins_ff_rand)):
            if bin_sums_ff_rand[b_idx] + size <= capacity:
                bins_ff_rand[b_idx].append(idx)
                bin_sums_ff_rand[b_idx] += size
                placed = True
                break
        
        if not placed:
            bins_ff_rand.append([idx])
            bin_sums_ff_rand.append(size)
            
    co_occurrence_counts += extract_co_occurrence(bins_ff_rand)
    
    # Combine co-occurrence with a "dead gap" penalty.
    # The directive suggests: inversely proportional to the count of items that fit in the residual space.
    # residual = capacity - (d_i + d_j)
    # If residual is small, few items fit -> high score.
    # If residual is large, many items fit -> low score (less attractive? or maybe we want to reward tight fits).
    # Actually, "rewarding pairs that create dead gaps" means high eta for small residuals.
    # Let's compute for each pair (i, j) the number of items k such that demands[k] <= residual.
    # We can do this efficiently with sorting and binary search or vectorization.
    
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Compute residual for all pairs
    residual = capacity - pair_sums
    
    # For each pair (i, j), count how many items fit in residual[i, j]
    # We want to reward pairs where few items fit in the residual (tight packing).
    # So, eta should be high when count is low.
    # Let's use: gap_score = 1.0 / (1.0 + count_fitting)
    
    # To compute count_fitting for all pairs:
    # For a given residual value r, count_fitting = number of demands <= r.
    # We can sort demands and use searchsorted.
    
    sorted_demands = np.sort(demands)
    
    # residual is (n, n). We need to count for each element how many sorted_demands are <= it.
    # searchsorted returns the index where the element would be inserted to maintain order.
    # For 'right' side, it gives the number of elements <= value.
    # But we need to do this for each element in residual.
    # np.searchsorted can handle arrays.
    # However, searchsorted operates on 1D arrays. We can flatten, search, and reshape.
    
    flat_residual = residual.flatten()
    # Only valid residuals are positive. Invalid pairs have residual < 0.
    # We only care about valid pairs.
    
    # For efficiency, we can just compute for all, but mask later.
    # Note: if residual < 0, count should be 0 (or irrelevant).
    
    # np.searchsorted(sorted_demands, flat_residual, side='right')
    # This returns an array of indices.
    
    # But wait, if residual is negative, searchsorted might return 0.
    # Let's compute counts.
    
    counts = np.searchsorted(sorted_demands, flat_residual, side='right')
    
    # Reshape back
    counts_matrix = counts.reshape(n, n)
    
    # For invalid pairs, counts might be non-zero if residual is negative?
    # If residual < 0, searchsorted(sorted_demands, negative, side='right') returns 0 because all demands are >= 20 > 0.
    # So counts_matrix will be 0 for invalid pairs.
    
    # Now, compute gap_score.
    # We want high score for low counts.
    # gap_score = 1.0 / (1.0 + counts_matrix)
    
    gap_score = 1.0 / (1.0 + counts_matrix)
    
    # Combine with co-occurrence.
    # The directive says "inversely proportional to the count... thereby explicitly rewarding pairs...".
    # It doesn't explicitly say to combine with co-occurrence, but the current design uses it.
    # Let's try multiplying or adding.
    # The current code uses: eta = 1.0 + co_occurrence_counts.
    # Let's try: eta = co_occurrence_counts * gap_score? Or eta = (1 + co_occurrence_counts) * gap_score?
    # Or maybe just gap_score?
    # The inspiration 1 (score 1.0530) used co-occurrence frequency prior.
    # The current program (0.9937) uses co-occurrence + 1.
    # Let's try blending: eta = (1.0 + co_occurrence_counts) * gap_score.
    # This rewards pairs that both co-occur AND have small residuals.
    
    # Let's experiment with the weighting.
    # If we just use gap_score, we lose the co-occurrence signal.
    # If we just use co_occurrence, we ignore the gap.
    # The directive specifically asks to investigate the "dead gap" strategy.
    
    # Let's set eta[valid] = (1.0 + co_occurrence_counts[valid]) * gap_score[valid]
    
    eta[valid] = (1.0 + co_occurrence_counts[valid]) * gap_score[valid]
    
    # Diagonal: self-pairing is invalid/meaningless in this context.
    np.fill_diagonal(eta, 1e-9)
    
    # Final check for finite values
    if not np.all(np.isfinite(eta)):
        eta = np.full((n, n), 1e-9)
        np.fill_diagonal(eta, 1e-9)
        
    return eta

# EVOLVE-BLOCK-END
