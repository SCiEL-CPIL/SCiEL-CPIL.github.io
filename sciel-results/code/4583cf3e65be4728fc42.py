import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # Returns list of counts for sums N_LOW..N_HIGH
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_arr = np.array(list(A), dtype=np.int32)
        n = len(A_arr)
        for i in range(n):
            ai = A_arr[i]
            # Vectorized calculation for j >= i
            # We need pairs (ai, aj) where j >= i
            # Sums are ai + A_arr[i:]
            sums = ai + A_arr[i:]
            # Filter sums in range
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            valid_sums = sums[mask]
            # np.bincount is fast for small integer ranges
            # We only need to add to the specific indices
            # Since valid_sums are in [128, 256], we can map to [0, 127]
            indices = valid_sums - N_LOW
            # Use np.add.at to handle duplicates in indices
            np.add.at(reps, indices, 1)
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    # Phase 1: Initialize a candidate set
    # To cover [128, 256] with sums of two elements from [0, 128]:
    # Min sum 128 requires e.g. 64+64 or 0+128.
    # Max sum 256 requires 128+128.
    # A random set of size ~50-60 often covers if spread well.
    # Let's try a random initialization with some structure.
    
    best_A = None
    best_peak = float('inf')
    
    # Initial random restart
    # Start with a set that is likely to cover: 
    # Include 0, 128, and random middle values.
    # Actually, just random sampling of size 60 from [0, 128] might miss edges.
    # Let's force include some boundary helpers.
    # 128 must be sum. 128 = 64+64 or 0+128 etc.
    # 256 must be sum. 256 = 128+128. So 128 MUST be in A.
    # 128 must be in A.
    
    # Let's try a few random initializations
    for _ in range(10):
        # Always include 128 to cover 256
        # Need to cover 128. If 128 is in A, 128 = 128+0. So 0 is helpful, or 64, etc.
        # Let's just pick a random set of size 50, force 128 in it.
        A_init = set(random.sample(range(0, 129), 50))
        A_init.add(128)
        A_init.add(0) # Helps cover lower end if 128 is there (128+0=128)
        
        reps = calc_reps(A_init)
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_init)
        else:
            # Even if not covered, we might use it as a starting point for repair?
            # For now, just keep best valid one.
            pass
            
    if best_A is None:
        # Fallback to seed if random init failed to cover (unlikely but possible)
        return list(range(64, 129))

    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a move: remove one element, add another
        # Or swap: remove one, add one (keep size constant)
        # Or add/remove to change size? 
        # Increasing size generally lowers peak (more ways to sum, but wait, 
        # more elements means MORE pairs, so peak might INCREASE if not careful.
        # Actually, to minimize peak multiplicity, we want to spread the sums.
        # Smaller set = fewer pairs = potentially lower peak, but harder to cover.
        # Larger set = more pairs = higher peak? 
        # Let's check: if A has size k, there are k(k+1)/2 pairs.
        # Total representations = k(k+1)/2.
        # Average rep per sum in [128,256] (129 sums) is k(k+1)/(2*129).
        # If k=50, avg ~ 50*51/258 ~ 9.9.
        # If k=100, avg ~ 100*101/258 ~ 39.
        # So SMALLER sets have lower average multiplicity.
        # We want the smallest k that covers the range, and distribute sums evenly.
        
        # Strategy: Try to remove an element if coverage is maintained (reduces k).
        # Or swap elements to balance counts.
        
        # Move Type 1: Swap (keep size same)
        # Move Type 2: Remove (decrease size) - only if still covered
        # Move Type 3: Add (increase size) - usually bad for peak, but might fix coverage
        
        # Let's focus on Swap and Remove.
        
        # Pick a random element to remove
        if len(current_A) == 0:
            break
            
        item_to_remove = random.choice(list(current_A))
        
        # Pick a random element to add (for swap) or just remove (for delete)
        # Let's do Swap first.
        new_item = random.randint(0, 128)
        
        # Apply Swap
        new_A = set(current_A)
        new_A.remove(item_to_remove)
        new_A.add(new_item)
        
        new_reps = calc_reps(new_A)
        
        if is_covered(new_reps):
            new_peak = get_peak(new_reps)
            delta_peak = new_peak - current_peak
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_peak < 0:
                current_A = new_A
                current_reps = new_reps
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)
            else:
                # Accept worse solution with probability
                prob = np.exp(-delta_peak / T)
                if random.random() < prob:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
        else:
            # Not covered, reject (or penalize heavily)
            pass
            
        T *= cooling_rate
        
        # Occasionally try to reduce size?
        # If we are stuck at a local optimum with high peak, maybe removing an element
        # that is "redundant" for coverage helps.
        # But calc_reps is O(k^2). Removing an element is O(k) to update reps?
        # No, we recalc.
        
        # Let's also try pure removal if we have been stuck?
        # For now, just swap is robust.
        
    # Final check
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
