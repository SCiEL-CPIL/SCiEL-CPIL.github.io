"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.
    
    Strategy:
    Instead of a static pair-based heuristic, we simulate a simple packing process
    (similar to Best-Fit Decreasing) to determine the 'context' of each item.
    Specifically, for each item j, we estimate the 'remaining space' in the bin 
    where it would likely be placed if items were packed in a greedy manner.
    
    However, the ACO heuristic H[i, j] is defined as the attractiveness of placing 
    j after i. In many bin packing ACO formulations, the heuristic can be 
    state-independent (precomputed) but should reflect the global structure.
    
    A common effective heuristic for bin packing is based on the 'complementarity' 
    of items. If item i is in a bin, the remaining space is R. Placing j is good 
    if j fits and leaves little space (good fit) or if j is small (flexibility).
    
    Since we must return a static matrix H[i, j] computed once, we can use a 
    proxy: 
    1. Sort items by size descending to simulate packing.
    2. For each item j, determine the 'target bin remaining space' it would 
       ideally fill. 
    3. The attractiveness of j after i should be high if i and j are 'complementary'.
    
    Actually, a simpler and often more robust strategy for static heuristics in 
    ACO bin packing is to favor transitions that lead to bins with higher 
    utilization. 
    Let's use a heuristic based on the inverse of the item size for the *next* 
    item, but weighted by the current item's size? No, that's just the seed.
    
    Let's look at the inspiration: 
    - p5 (0.98) used 1/demands[j] (favor small items). This is a column-wise constant.
    - p4 (1.01) had a complex change, score improved.
    
    Let's try a heuristic that encourages filling bins completely.
    If item i has size s_i, and we place item j with size s_j, the 'waste' 
    created or remaining depends on the bin. But we don't know the bin.
    
    Alternative Strategy: 
    Use a heuristic that favors placing items of similar size together (clustering) 
    or complementary sizes. 
    In bin packing, "complementary" usually means s_i + s_j is close to capacity 
    (for 2-item bins) or just fits well.
    
    Let's try: H[i, j] = 1 / (1 + |s_i + s_j - capacity/2|) ? No.
    
    Let's go with a "Remaining Space" approximation.
    If we assume items are packed in descending order of size (a common heuristic 
    baseline), we can simulate the packing to find out which bin each item ends up 
    in, and what the remaining space was when it was placed.
    Then, H[i, j] can be high if j's size is close to the remaining space of the 
    bin that i is in.
    
    However, H[i, j] is "attractiveness of j after i". 
    If i is in bin B_i, and the remaining space in B_i is R_i, then placing j 
    is attractive if s_j <= R_i and s_j is close to R_i (best fit).
    
    So:
    1. Sort indices by demand descending.
    2. Simulate First-Fit-Decreasing (FFD) or Best-Fit-Decreasing (BFD). Let's use BFD as it often yields better packing.
    3. For each item i, record the remaining space in its bin *before* i was placed? 
       No, "after i". So, if i is placed in bin B, the remaining space *after* 
       placing i is R_after_i.
    4. For a pair (i, j), the attractiveness of placing j after i should depend on 
       whether j fits in the remaining space of i's bin.
       But wait, j might not be placed in i's bin in the simulation.
       
    The ACO transition is: current item is i, choose next item j.
    The heuristic H[i, j] should guide the ant to pick j.
    If H[i, j] is high when s_j fits well in the remaining space of i's bin, 
    it encourages the ant to do the same.
    
    Let's compute:
    - Assign each item to a bin using BFD.
    - For each item i, calculate `rem_space[i]` = capacity - sum of items in i's bin (including i).
    - H[i, j] = 1.0 if demands[j] <= rem_space[i], else 1e-9.
    - To make it more nuanced, H[i, j] = 1.0 / (1.0 + abs(rem_space[i] - demands[j])) if fits, else 1e-9.
      This favors "best fit" (smallest remaining space after placement).
      
    This creates a matrix where H[i, j] is high if j is a good "next" item for the 
    bin containing i. This is a state-aware heuristic encoded statically.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # 1. Sort items by size descending to simulate packing
    sorted_indices = np.argsort(demands)[::-1]
    sorted_demands = demands[sorted_indices]
    
    # 2. Simulate Best-Fit-Decreasing (BFD)
    # bins: list of lists of item indices (original indices)
    bins = []
    bin_remaining = []
    item_bin_idx = np.zeros(n, dtype=int) # which bin each item is in
    item_pos_in_bin = np.zeros(n, dtype=int) # not strictly needed, but good for tracking
    
    for idx in sorted_indices:
        size = demands[idx]
        best_bin = -1
        best_remaining = np.inf
        
        for b_idx in range(len(bins)):
            if bin_remaining[b_idx] >= size:
                # Best fit: smallest remaining space that fits
                if bin_remaining[b_idx] < best_remaining:
                    best_remaining = bin_remaining[b_idx]
                    best_bin = b_idx
        
        if best_bin == -1:
            # Open new bin
            best_bin = len(bins)
            bins.append([])
            bin_remaining.append(capacity - size)
        else:
            bin_remaining[best_bin] -= size
        
        bins[best_bin].append(idx)
        item_bin_idx[idx] = best_bin
    
    # 3. Calculate remaining space for each item's bin AFTER the item is placed
    # We need the total sum in the bin to know the final remaining space.
    # Actually, the "state" after placing i is the remaining space in that bin.
    # Since the heuristic is static, we use the final state of the bin from the simulation.
    # This might be too strong (it assumes the bin is full), but it guides towards 
    # items that fit into the "holes" created by the packing.
    
    bin_final_remaining = np.zeros(len(bins))
    for b_idx in range(len(bins)):
        bin_final_remaining[b_idx] = bin_remaining[b_idx]
    
    # For each item i, rem_space[i] = bin_final_remaining[item_bin_idx[i]]
    rem_space = bin_final_remaining[item_bin_idx]
    
    # 4. Construct Heuristic Matrix H[i, j]
    # H[i, j] is attractive if j fits in the remaining space of i's bin.
    # We prefer j if it is a "best fit" for that remaining space.
    # H[i, j] = 1 / (1 + |rem_space[i] - demands[j]|) if demands[j] <= rem_space[i], else 1e-9
    
    # Vectorized computation
    # rem_space is (n,), demands is (n,)
    # rem_space[:, None] is (n, 1)
    # demands[None, :] is (1, n)
    
    rs = rem_space[:, None]
    ds = demands[None, :]
    
    # Check fit
    fits = ds <= rs
    
    # Calculate attractiveness
    # If fits, 1 / (1 + abs(rs - ds)), else 1e-9
    diff = np.abs(rs - ds)
    attr = 1.0 / (1.0 + diff)
    
    H = np.where(fits, attr, 1e-9)
    
    # Ensure no negative or zero (attr is always > 0)
    # The task says non-positive become 1e-9. Our attr is positive.
    
    # Check time
    if time.time() > deadline:
        # Fallback to a simple heuristic if simulation was too slow (unlikely for n=120)
        return np.ones((n, n))
    
    return H
# EVOLVE-BLOCK-END
