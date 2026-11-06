import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s/90s limit

    # 1. Generate all arithmetic progressions (APs)
    # An AP is defined by (start, step). Length is implicit (max possible within [0, n)).
    # We need to consider all starts and steps such that at least one element exists.
    # Actually, the problem says "contained in [1, 256]". Indices 0..255 correspond to 1..256.
    # AP: a, a+s, a+2s, ... < n.
    # We need to check sums for ALL prefixes of these APs? 
    # "Minimize the maximum absolute signed sum on ANY arithmetic progression ... and length including one."
    # This implies we must check every contiguous subsequence of an AP? 
    # No, an arithmetic progression IS a specific sequence. 
    # Usually "sum on an AP" means the sum of the terms in that specific AP.
    # But "length including one" suggests that for a given start and step, we might have multiple lengths?
    # "ANY arithmetic progression contained in [1,256]".
    # An AP is typically defined by start, step, and length.
    # So we need to consider ALL (start, step, length) tuples where the AP fits in [0, n).
    # This is a huge number of APs.
    # Total APs:
    # For step s:
    #   Starts a can be 0..n-1.
    #   Max length L_max(a, s) = floor((n-1-a)/s) + 1.
    #   Number of APs for this (a,s) is L_max.
    #   Total for step s is sum_{a=0}^{n-1} (floor((n-1-a)/s) + 1).
    #   This is roughly n^2/(2s).
    #   Sum over s=1..n is n^2/2 * H_n ~ 256^2 * 3 / 2 ~ 100k.
    #   Wait, let's calculate exactly.
    #   For s=1: sum_{a=0}^{255} (256-a) = 256*257/2 = 32896.
    #   For s=2: sum_{a=0}^{255} (floor(255-a)/2 + 1) ~ 256 * 64 = 16384.
    #   Total is roughly n^2/2 * ln(n) + ...
    #   Let's just generate them. It's manageable.
    
    # However, checking every single AP sum for every flip is expensive if we recompute from scratch.
    # We can use the "aps_containing" trick but it's even larger.
    # Alternative: 
    # Note that an AP sum is a linear combination of signs.
    # We can precompute the list of APs as arrays of indices.
    # Then for each flip, we update the sums of all APs containing that index.
    
    # Let's generate the APs.
    ap_indices = []
    for s in range(1, n + 1):
        for a in range(n):
            # Generate all lengths for this start and step
            # Max index is n-1.
            # Sequence: a, a+s, ...
            # We iterate length from 1 to max_len
            # It's more efficient to store the full list of indices for the max length,
            # and then consider all prefixes?
            # No, an AP of length L is a specific set.
            # If we store the max length AP, we can compute prefix sums.
            # But the "sum on ANY AP" means we take the sum of the signs at the positions of that AP.
            # If we have the list of indices for the max AP, say idx_list = [i0, i1, ..., ik].
            # Then the APs starting at a with step s are:
            # [i0], [i0, i1], [i0, i1, i2], ...
            # So we just need to track the running sum?
            # But the signs change.
            
            # Let's just store every distinct AP as a list of indices.
            # Memory: ~100k APs * avg length ~ 10 = 1M ints. Fine.
            
            current = a
            while current < n:
                # This is not efficient to append one by one to a global list of lists
                # Let's collect indices for this (a,s) first
                pass
            
            # Better:
            idxs = []
            cur = a
            while cur < n:
                idxs.append(cur)
                cur += s
            
            # Now idxs is the max AP.
            # The APs are prefixes of idxs.
            # We can store idxs and know that all prefixes are APs.
            # To update sums, if we flip index i, it affects:
            # 1. All APs (a', s') such that i is in the prefix.
            # This is equivalent to: for every AP that contains i, update its sum.
            
            # Let's stick to the previous method: generate all APs as separate lists.
            for l in range(len(idxs)):
                ap_indices.append(idxs[:l+1])
                
    # This loop is O(n^2 log n) which might be slow in Python.
    # Let's optimize generation.
    ap_indices = []
    for s in range(1, n + 1):
        for a in range(n):
            idxs = list(range(a, n, s))
            for l in range(1, len(idxs) + 1):
                ap_indices.append(idxs[:l])
                
    num_aps = len(ap_indices)
    # Convert to numpy arrays for vectorized sum updates?
    # Or just keep as lists of lists and use a sparse update strategy.
    
    # Precompute: for each index i, which APs contain it?
    # aps_containing[i] = list of AP indices j such that i in ap_indices[j]
    aps_containing = [[] for _ in range(n)]
    for j, ap in enumerate(ap_indices):
        for i in ap:
            aps_containing[i].append(j)
            
    # Convert to numpy arrays for fast access
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Initialize signs with Thue-Morse (good initial distribution)
    def thue_morse(i):
        return 1 if bin(i).count('1') % 2 == 0 else -1
        
    signs = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Initialize AP sums
    # sums[j] = sum(signs[ap_indices[j]])
    # Doing this naively is O(total_size_of_aps) ~ 1M operations. Fast.
    sums = np.zeros(num_aps, dtype=np.int32)
    for j, ap in enumerate(ap_indices):
        sums[j] = np.sum(signs[ap])
        
    current_max = np.max(np.abs(sums))
    
    # Local Search with Simulated Annealing or Hill Climbing
    # Given the large number of APs, checking np.max(np.abs(sums)) is O(num_aps) ~ 100k.
    # This is the bottleneck.
    # We can maintain a histogram of absolute sums?
    # Sums can range from -256 to 256.
    # Let's use a histogram of the absolute values.
    # hist[k] = number of APs with |sum| == k.
    # Then max is the largest k with hist[k] > 0.
    
    def update_hist(sums, i, old_val, new_val):
        # This is tricky because we only change a few sums.
        # Let's just recompute max if the number of flips is small?
        # No, we flip one sign at a time.
        # We can update the histogram incrementally.
        
        pass
        
    # Actually, 100k operations for np.max is very fast in NumPy (microseconds).
    # So we can just use np.max.
    
    # Algorithm:
    # 1. Hill climb: flip sign if it reduces current_max.
    # 2. If stuck, do a random perturbation (accept worse move) and continue.
    # 3. Keep track of best solution found.
    
    best_signs = signs.copy()
    best_max = current_max
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing style
    # T starts high, decreases.
    # But we have a hard deadline.
    # Let's do a loop of "passes".
    # In each pass, try to flip each index once.
    
    while time.time() < deadline:
        improved = False
        # Shuffle order to avoid bias
        order = rng.permutation(n)
        
        for i in order:
            if time.time() > deadline:
                break
                
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Update sums
            # delta = new_sign - old_sign = -2 * old_sign
            delta = -2 * old_sign
            affected_aps = aps_containing_np[i]
            sums[affected_aps] += delta
            
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                # Accept
                signs[i] = new_sign
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Reject (or accept with prob exp(-delta/T) for SA, but let's stick to hill climb first)
                # Revert
                sums[affected_aps] -= delta
                # signs[i] is already old_sign, so no change needed there
            
        if not improved:
            # Stuck. Perturb.
            # Flip a random number of bits
            num_flips = rng.integers(1, 10)
            for _ in range(num_flips):
                i = rng.integers(n)
                old_sign = signs[i]
                new_sign = -old_sign
                delta = -2 * old_sign
                affected_aps = aps_containing_np[i]
                sums[affected_aps] += delta
                signs[i] = new_sign
                
            # Recalculate max
            current_max = np.max(np.abs(sums))
            
            # If this perturbation led to a better max, great. If not, we continue from here.
            # We don't revert the perturbation, we accept it as a new starting point.
            # This allows escaping local minima.
            
    return [int(s) for s in best_signs]
