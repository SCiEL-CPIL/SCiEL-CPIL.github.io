"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix combining:
    1. Data-driven co-occurrence prior from Randomized FFD simulations.
    2. Structural gap-utility: rewards pairs that leave residual space too small
       for any remaining item (creating "dead gaps" that cannot be filled).
    
    The final heuristic is a geometric blend of the co-occurrence score and
    the gap-utility score, both normalized to [1, large] for valid pairs.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin for ACO and overhead
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # --- Part 1: Co-occurrence prior from Randomized FFD ---
    C = np.zeros((n, n), dtype=np.int32)
    
    num_sims = 2000
    
    for sim in range(num_sims):
        if time.time() > deadline:
            break
            
        noise = np.random.rand(n)
        sort_keys = np.lexsort((noise, -demands))
        
        bins = []
        bin_loads = []
        
        for idx in sort_keys:
            d = demands[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + d <= capacity:
                    bins[b].append(idx)
                    bin_loads[b] += d
                    placed = True
                    break
            
            if not placed:
                bins.append([idx])
                bin_loads.append(d)
        
        for b in bins:
            if len(b) > 1:
                for i in range(len(b)):
                    for j in range(i+1, len(b)):
                        u = b[i]
                        v = b[j]
                        C[u, v] += 1
                        C[v, u] += 1
                        
    # --- Part 2: Gap-utility score ---
    # For pair (i, j), residual = capacity - (d_i + d_j)
    # Count how many items k (k != i, k != j) satisfy d_k <= residual
    # Fewer such items means the gap is "dead" -> higher utility
    # If residual < min(demands), no item fits -> maximum utility
    
    min_demand = np.min(demands)
    
    # Compute residual for all pairs
    pair_sum = demands[:, None] + demands[None, :]
    residual = capacity - pair_sum
    
    # For each pair (i,j), count items k where demands[k] <= residual[i,j] and k != i, k != j
    # This is O(n^3) if done naively, but we can vectorize partially
    # For each unique residual value, we can precompute counts
    
    # Actually, let's compute this more efficiently:
    # For each item k, it fits in residual r if demands[k] <= r
    # So for pair (i,j), the count of fitting items (excluding i and j) is:
    #   count_all = number of items with demand <= residual[i,j]
    #   minus 1 if demands[i] <= residual[i,j] (i.e., d_i <= capacity - d_i - d_j)
    #   minus 1 if demands[j] <= residual[i,j]
    
    # Precompute sorted demands for binary search
    sorted_demands = np.sort(demands)
    
    # For each pair (i,j), compute count of items with demand <= residual[i,j]
    # Using np.searchsorted
    counts_all = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            r = residual[i, j]
            if r < 0:
                counts_all[i, j] = 0
            else:
                # Number of items with demand <= r
                counts_all[i, j] = np.searchsorted(sorted_demands, r, side='right')
    
    # Now subtract self if it fits
    # Item i fits in residual[i,j] if demands[i] <= residual[i,j]
    # i.e., d_i <= capacity - d_i - d_j, i.e., 2*d_i + d_j <= capacity
    # Similarly for j
    
    # For pair (i,j), the count of OTHER items that fit is:
    # counts_all[i,j] - (1 if d_i <= residual[i,j] else 0) - (1 if d_j <= residual[i,j] else 0)
    # But we need to be careful: if i == j, we only subtract once, but diagonal is invalid anyway
    
    # Let's compute this vectorized
    # d_i <= residual[i,j] means demands[i] <= capacity - demands[i] - demands[j]
    # i.e., 2*demands[i] + demands[j] <= capacity
    
    i_fits = (2.0 * demands[:, None] + demands[None, :]) <= capacity
    j_fits = (demands[:, None] + 2.0 * demands[None, :]) <= capacity
    
    # Note: i_fits[i,j] is True if item i fits in residual of pair (i,j)
    # j_fits[i,j] is True if item j fits in residual of pair (i,j)
    
    # The count of other items that fit:
    other_fit_counts = counts_all - i_fits.astype(np.int32) - j_fits.astype(np.int32)
    other_fit_counts = np.maximum(other_fit_counts, 0)
    
    # Gap utility: inversely proportional to other_fit_counts
    # If other_fit_counts == 0, the gap is dead -> high utility
    # If other_fit_counts is large, the gap is fillable -> low utility
    # Use: utility = 1.0 / (1.0 + other_fit_counts)
    # This gives 1.0 when 0 items fit, 0.5 when 1 fits, 0.33 when 2 fit, etc.
    
    gap_utility = 1.0 / (1.0 + other_fit_counts.astype(float))
    
    # --- Combine the two signals ---
    valid_mask = (pair_sum <= capacity) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    
    # Co-occurrence score
    max_c = np.max(C)
    if max_c == 0:
        co_occ_score = np.ones((n, n), dtype=float)
    else:
        # Normalize C to [0, 1] then map to [1, large]
        co_occ_norm = C.astype(float) / max_c
        # Use exponential to amplify differences
        co_occ_score = 1.0 + np.exp(co_occ_norm * 3.0)
    
    # Gap utility is already in [0, 1], map to [1, large]
    gap_score = 1.0 + gap_utility * 5.0
    
    # Geometric blend: eta = (co_occ_score * gap_score) ** 0.5
    # Or arithmetic: eta = 0.5 * co_occ_score + 0.5 * gap_score
    # Let's try geometric mean to balance scales
    eta = np.sqrt(co_occ_score * gap_score)
    
    # Apply only to valid pairs, set invalid to 1e-9
    eta_final = np.full((n, n), 1e-9, dtype=float)
    eta_final[valid_mask] = eta[valid_mask]
    
    # Ensure diagonal is low
    np.fill_diagonal(eta_final, 1e-9)
    
    return eta_final
# EVOLVE-BLOCK-END
