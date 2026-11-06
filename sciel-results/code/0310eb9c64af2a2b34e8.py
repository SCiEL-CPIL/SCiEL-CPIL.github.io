import time
import random
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Greedy Set Cover: Build a valid set A quickly by iteratively adding the element
       that covers the most "uncovered" target sums [128, 256].
    2. Multi-Start Local Search: From the greedy start, perform local search using
       2-swap and 3-swap moves to minimize the peak multiplicity.
    3. Fast Evaluation: Use NumPy convolution to compute representation counts.
    """
    deadline = time.time() + 85
    rng = random.Random(42)
    
    # Precompute target sums
    target_sums = np.arange(128, 257)
    n_targets = len(target_sums)
    
    # Fast evaluation function
    def evaluate(A_set):
        """
        Returns (peak_multiplicity, is_valid).
        Uses convolution to count sums.
        """
        if not A_set:
            return float('inf'), False
        
        # Create boolean array for A
        A_bool = np.zeros(129, dtype=np.int32)
        for a in A_set:
            if 0 <= a <= 128:
                A_bool[a] = 1
            else:
                return float('inf'), False # Invalid element
        
        # Convolution of A with A gives counts of ordered pairs (i,j) such that A[i]+A[j]=k
        conv = np.convolve(A_bool, A_bool, mode='full')
        
        # Extract counts for target sums [128, 256]
        ordered_counts = conv[128:257].copy()
        
        # Convert ordered counts to unordered pairs with replacement
        # Formula: count_unordered(k) = (count_ordered(k) + (1 if k%2==0 and k//2 in A else 0)) / 2
        even_mask = (target_sums % 2 == 0)
        
        if np.any(even_mask):
            mids = target_sums[even_mask] // 2
            # mids are in range [64, 128]
            # Check if mid is in A
            # We need to be careful with indices. A_bool is length 129 (0..128).
            # mids are valid indices for A_bool.
            diagonal_present = A_bool[mids]
            ordered_counts[even_mask] += diagonal_present
        
        final_counts = ordered_counts // 2
        
        # Check validity: all counts must be > 0
        if np.any(final_counts == 0):
            return float('inf'), False
            
        peak = np.max(final_counts)
        return int(peak), True

    # --- Phase 1: Greedy Set Cover ---
    # Start with an empty set? No, start with a few random or strategic elements.
    # To ensure we can cover 128 and 256, we need elements that can sum to these.
    # 128 = 0+128, 64+64, etc.
    # 256 = 128+128. So 128 must be in A.
    
    A = set()
    A.add(128)
    
    # Track coverage
    # We want to add elements that maximize the number of new target sums covered.
    # A sum s is covered if there exists a,b in A such that a+b=s.
    
    # Precompute potential contributions?
    # For a new element x, it forms pairs (x, a) for all a in A.
    # Sums are x + a.
    
    current_counts = np.zeros(n_targets, dtype=np.int32)
    
    # Update counts for initial A={128}
    # 128+128 = 256.
    # Index for 256 in target_sums: 256-128 = 128.
    current_counts[128] += 1
    
    # Greedy loop
    max_elements = 50 # Safety bound
    while True:
        # Check if valid
        if np.all(current_counts > 0):
            break
        
        # Find the element x in [0, 128] \ A that adds the most coverage
        best_x = -1
        best_gain = 0
        best_new_counts = None
        
        # To speed this up, we can sample candidates or iterate.
        # 129 elements is small enough to iterate all.
        # For each candidate x:
        #   new_sums = x + a for a in A
        #   gain = number of sums in new_sums that are currently 0 (or low?)
        #   Actually, we just need to cover 0s. Minimizing peak is phase 2.
        #   So gain = count of s in new_sums such that current_counts[s] == 0.
        
        # Optimization: Precompute A list
        A_list = list(A)
        
        for x in range(129):
            if x in A:
                continue
            
            # Calculate new sums
            # sums = x + np.array(A_list)
            # We only care about sums in [128, 256]
            # And we only care if they are currently 0.
            
            # Vectorized check
            # Create array of sums
            sums = x + np.array(A_list, dtype=np.int32)
            # Filter valid range
            valid_mask = (sums >= 128) & (sums <= 256)
            if not np.any(valid_mask):
                continue
            
            valid_sums = sums[valid_mask]
            # Map to indices in current_counts
            # Index = sum - 128
            indices = valid_sums - 128
            
            # Count how many of these indices have current_counts == 0
            # unique_indices = np.unique(indices)
            # gain = np.sum(current_counts[unique_indices] == 0)
            
            # Simpler: just sum over the specific indices (duplicates count as 1 coverage for "is it zero"?)
            # Wait, if x+a1 = x+a2, that's the same sum.
            # We want to know if the sum is *newly covered*.
            # So unique sums matter.
            
            unique_indices = np.unique(indices)
            gain = np.sum(current_counts[unique_indices] == 0)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                # We don't need to store best_new_counts yet, only if we pick it.
                
        if best_x == -1 or best_gain == 0:
            # Stuck? This shouldn't happen if we start with 128 and go up to 50 elements.
            # Fallback: add random
            candidates = [x for x in range(129) if x not in A]
            if not candidates:
                break
            best_x = rng.choice(candidates)
            
        # Add best_x
        A.add(best_x)
        A_list.append(best_x)
        
        # Update current_counts
        # New sums formed by (best_x, a) for all a in A (including best_x itself? Yes, pairs are unordered, so (x,x) is valid)
        # Actually, my loop above considered x + a for a in OLD A.
        # Now I need to add contributions from (x, x) and (x, a) for a in OLD A.
        # The loop above calculated gain based on x + OLD_A.
        # It missed (x, x).
        # Let's just recompute counts for the new A to be safe and simple, since |A| is small.
        # Or update incrementally:
        # 1. Add pairs (x, a) for a in OLD A.
        # 2. Add pair (x, x).
        
        # Incremental update:
        old_A_list = A_list[:-1]
        
        # Pairs (x, a) for a in old_A_list
        sums_old = best_x + np.array(old_A_list, dtype=np.int32)
        valid_mask_old = (sums_old >= 128) & (sums_old <= 256)
        if np.any(valid_mask_old):
            idx_old = sums_old[valid_mask_old] - 128
            # Increment counts
            # np.bincount is good here
            # But we just need to increment specific indices.
            # Since duplicates are possible, bincount handles it.
            # However, current_counts is a fixed array.
            # We can use np.add.at or just loop if small.
            # |A| is small (<50).
            for idx in idx_old:
                current_counts[idx] += 1
                
        # Pair (x, x)
        sum_xx = 2 * best_x
        if 128 <= sum_xx <= 256:
            idx_xx = sum_xx - 128
            current_counts[idx_xx] += 1
            
        # Check time
        if time.time() > deadline - 20:
            break
            
        if len(A) > max_elements:
            break

    # Ensure we have a valid set
    # If greedy didn't finish, force validity
    if not np.all(current_counts > 0):
        # Fallback: Add random elements until valid
        while not np.all(current_counts > 0) and time.time() < deadline - 10:
            x = rng.randint(0, 128)
            if x not in A:
                A.add(x)
                # Recompute counts? Or update.
                # Just recompute for safety in this fallback
                peak, is_valid = evaluate(A)
                if is_valid:
                    # Update current_counts to match?
                    # We can just break and let Phase 2 handle it.
                    break
            else:
                # If x in A, try again
                pass
        
        # If still invalid, return a known valid set
        # range(64, 129) is valid.
        if not evaluate(A)[1]:
            return list(range(64, 129))

    # --- Phase 2: Local Search ---
    
    current_A = set(A)
    current_peak, _ = evaluate(current_A)
    best_A = set(current_A)
    best_peak = current_peak
    
    # Multi-start with different random perturbations?
    # Or just run local search from the greedy start.
    # Given time constraints, one good run of local search is better than many bad ones.
    
    # Moves:
    # 1. 2-Swap: Remove x, Add y. (Size constant)
    # 2. 3-Swap: Remove x, Remove z, Add y. (Size decreases by 1)
    # 3. 1-Add: Add y. (Size increases by 1)
    
    # We want to minimize peak.
    # If we remove an element, peak might decrease or increase (if it was critical).
    # If we add an element, peak might increase.
    # So 2-Swap is the most balanced.
    
    all_elements = list(range(129))
    
    # Precompute A list for speed
    A_list = list(current_A)
    
    # Local Search Loop
    while time.time() < deadline - 5:
        
        # Generate a random 2-swap move
        # Pick x to remove, y to add
        # To make it smart, pick x that is "high cost" (appears in many high-count sums)
        # and y that is "low cost" or fills a gap?
        # For now, random.
        
        # Optimization: Try to pick x from A and y from [0,128]\A
        # To speed up, we can try a few random swaps and keep the best.
        
        best_neighbor_peak = current_peak
        best_neighbor_A = None
        
        # Try 5 random swaps
        for _ in range(5):
            if len(current_A) < 2:
                break
            x = rng.choice(list(current_A))
            # Pick y not in A
            # Random sampling for y
            y = rng.randint(0, 128)
            if y == x or y in current_A:
                continue
                
            neighbor_A = set(current_A)
            neighbor_A.remove(x)
            neighbor_A.add(y)
            
            # Evaluate
            # To save time, we can do a quick check?
            # Full evaluation is ~50us. 5 swaps = 250us.
            # 60s / 250us = 240,000 iterations. Plenty.
            
            new_peak, is_valid = evaluate(neighbor_A)
            
            if not is_valid:
                # Penalty for invalid
                # We can accept invalid states if they are "close" to valid?
                # No, let's stick to valid states for simplicity, or use a heavy penalty.
                # If we allow invalid, we need a good heuristic.
                # Let's just reject invalid.
                continue
                
            if new_peak < best_neighbor_peak:
                best_neighbor_peak = new_peak
                best_neighbor_A = neighbor_A
                
        # If we found a better neighbor, move to it
        if best_neighbor_A is not None and best_neighbor_peak < current_peak:
            current_A = best_neighbor_A
            current_peak = best_neighbor_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
        else:
            # Stuck? Try a random restart or a different move type.
            # Let's try a 3-swap (remove 2, add 1) to reduce size, which might lower peak.
            if len(current_A) > 2 and rng.random() < 0.1:
                # Remove 2 random, add 1 random
                x1, x2 = rng.sample(list(current_A), 2)
                y = rng.randint(0, 128)
                if y not in current_A:
                    neighbor_A = set(current_A)
                    neighbor_A.remove(x1)
                    neighbor_A.remove(x2)
                    neighbor_A.add(y)
                    new_peak, is_valid = evaluate(neighbor_A)
                    if is_valid and new_peak < current_peak:
                        current_A = neighbor_A
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(current_A)
            # Or just random walk?
            # If stuck, maybe we should add an element?
            # Adding elements increases density, likely increasing peak.
            # Removing elements decreases density, likely decreasing peak but risking validity.
            
    return list(best_A)

# EVOLVE-BLOCK-END
