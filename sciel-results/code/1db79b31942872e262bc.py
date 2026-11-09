import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    # Precompute pair indices for unordered pairs (i <= j)
    # We will maintain a binary mask of A and compute reps via matrix multiplication
    # A is a subset of [0, 128]. Let x be a vector of length 129 where x[i]=1 if i in A.
    # Sum matrix S = x^T x (outer product). S[i,j] = 1 if i,j in A.
    # We need counts of sums.
    # For a specific sum s, count = sum_{i+j=s, i<=j} S[i,j].
    
    # Efficient calculation using FFT?
    # The number of pairs (i,j) with i,j in [0,128] and i+j in [128,256] is small.
    # Total possible pairs in [0,128]^2 is 129^2 ~ 16641.
    # Target sums 128..256.
    
    # Let's use a direct method that is fast:
    # Precompute for each sum s in [128, 256] the list of pairs (i, j) with i<=j, i+j=s, i,j in [0,128].
    # Then reps[s] = sum( mask[i] * mask[j] for (i,j) in pairs[s] ).
    # This is O(K * N_pairs_per_sum).
    # N_pairs_per_sum is at most ~64.
    # K=129.
    # Total ops ~ 129 * 64 = ~8000 multiplications/additions.
    # This is very fast in Python if vectorized or even in a tight loop.
    
    # Let's precompute the pairs for each sum.
    pairs_for_sum = [None] * (N_HIGH + 1)
    for s in range(N_LOW, N_HIGH + 1):
        pairs = []
        # i ranges from max(0, s-128) to min(128, s//2)
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128 and j >= 0:
                pairs.append((i, j))
        pairs_for_sum[s] = np.array(pairs, dtype=np.int32).flatten() if pairs else np.array([], dtype=np.int32)

    def calc_reps(mask):
        # mask is a numpy array of 0/1 of length 129
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue
            # pairs is flattened: [i1, j1, i2, j2, ...]
            # We need sum(mask[i] * mask[j])
            # Reshape to (-1, 2)
            p = pairs.reshape(-1, 2)
            # Get mask values
            m_vals = mask[p]
            # Multiply pairs
            prods = m_vals[:, 0] * m_vals[:, 1]
            reps[idx] = np.sum(prods)
        return reps

    def get_objective(reps):
        uncovered_count = np.sum(reps == 0)
        if uncovered_count == 0:
            peak = int(np.max(reps))
            return peak
        else:
            return 10000 + uncovered_count * 100

    def mask_to_list(mask):
        return [i for i in range(129) if mask[i] == 1]

    # Phase 1: Initialization
    # Start with a random valid-ish set or a known good structure
    # A = [0, 128] + random subset of [1, 127]
    
    best_A = None
    best_obj = float('inf')
    best_mask = None
    
    # Try a few initializations
    for _ in range(10):
        mask = np.zeros(129, dtype=np.int32)
        # Force 0 and 128
        mask[0] = 1
        mask[128] = 1
        # Add random elements
        k = random.randint(30, 60)
        candidates = list(range(1, 128))
        random.shuffle(candidates)
        for i in range(k):
            mask[candidates[i]] = 1
        
        reps = calc_reps(mask)
        obj = get_objective(reps)
        
        if obj < best_obj:
            best_obj = obj
            best_mask = mask.copy()
            best_A = mask_to_list(mask)

    if best_mask is None:
        # Fallback
        mask = np.zeros(129, dtype=np.int32)
        for i in range(64, 129):
            mask[i] = 1
        return mask_to_list(mask)

    current_mask = best_mask.copy()
    current_reps = calc_reps(current_mask)
    current_obj = get_objective(current_reps)

    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate move
        # 1. Flip a random bit
        # 2. Swap two bits (remove one, add another)
        
        new_mask = current_mask.copy()
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Flip
            i = random.randint(0, 128)
            new_mask[i] = 1 - new_mask[i]
        elif move_type < 0.8:
            # Swap: pick one present, one absent, swap
            present = np.where(current_mask == 1)[0]
            absent = np.where(current_mask == 0)[0]
            if len(present) == 0 or len(absent) == 0:
                continue
            i = random.choice(present)
            j = random.choice(absent)
            new_mask[i] = 0
            new_mask[j] = 1
        else:
            # Add/Remove
            if random.random() < 0.5:
                # Add
                absent = np.where(current_mask == 0)[0]
                if len(absent) > 0:
                    j = random.choice(absent)
                    new_mask[j] = 1
            else:
                # Remove
                present = np.where(current_mask == 1)[0]
                if len(present) > 1: # Keep at least one?
                    i = random.choice(present)
                    new_mask[i] = 0

        if np.array_equal(new_mask, current_mask):
            continue

        new_reps = calc_reps(new_mask)
        new_obj = get_objective(new_reps)
        
        delta_obj = new_obj - current_obj
        
        accept = False
        if delta_obj < 0:
            accept = True
        else:
            # Avoid overflow
            if delta_obj / T > 100:
                prob = 0.0
            else:
                prob = np.exp(-delta_obj / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_mask = new_mask
            current_reps = new_reps
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_mask = current_mask.copy()
                best_A = mask_to_list(best_mask)
                
        T *= cooling_rate

    # Final validation and repair if necessary
    final_reps = calc_reps(best_mask)
    if np.any(final_reps == 0):
        # Greedy repair
        # Find missing sums
        missing_sums = [s for s in range(N_LOW, N_HIGH + 1) if final_reps[s - N_LOW] == 0]
        
        # For each missing sum, try to add elements to cover it
        # This is a set cover problem, but we can do a simple greedy:
        # For each missing sum s, pick a pair (i,j) that minimizes the increase in peak?
        # Or just pick a random pair and add i,j to A.
        
        # Let's try to repair by adding elements
        repair_mask = best_mask.copy()
        repair_reps = final_reps.copy()
        
        for s in missing_sums:
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue
            # Find a pair (i,j) that is "cheapest" to add
            # Cost: increase in reps for other sums
            # Simple heuristic: pick a pair where i,j are already in A? No, they aren't if reps is 0.
            # If reps[s]==0, no pair (i,j) with i,j in A sums to s.
            # We need to add at least one of i or j for some pair.
            # Let's pick a random pair and add both? Or just one?
            # If we add i, it might cover s if j is already in A? No, j is not in A for any pair summing to s?
            # Wait, if j is in A and i is not, then (i,j) contributes 0.
            # If we add i, and j is in A, then (i,j) becomes 1, covering s.
            # So for each pair (i,j), check if one is already in A.
            # If yes, adding the other is "cheap" (only affects sums involving the new element).
            # If neither is in A, adding both is "expensive".
            
            best_pair = None
            best_cost = float('inf')
            
            for idx in range(len(pairs) // 2):
                i = pairs[2*idx]
                j = pairs[2*idx+1]
                i_in = repair_mask[i]
                j_in = repair_mask[j]
                
                if i_in and j_in:
                    # Should not happen if reps[s]==0, unless bug
                    continue
                    
                if i_in or j_in:
                    # Cost 1: add one element
                    # Which one to add?
                    # Let's just add the one that is not in A
                    if not i_in:
                        new_elem = i
                    else:
                        new_elem = j
                        
                    # Estimate cost: how many sums does new_elem affect?
                    # Sums involving new_elem: new_elem + k for k in A.
                    # This is O(|A|).
                    # For simplicity, just add it.
                    best_pair = new_elem
                    best_cost = 1
                    break # Take the first one found
                    
            if best_pair is None:
                # No pair has one element in A.
                # Pick a random pair and add both?
                # Or pick a random element from the list of all i's in pairs?
                # Let's pick a random pair and add both.
                idx = random.randint(0, len(pairs)//2 - 1)
                i = pairs[2*idx]
                j = pairs[2*idx+1]
                repair_mask[i] = 1
                repair_mask[j] = 1
            else:
                repair_mask[best_pair] = 1
            
            # Update reps? This is expensive.
            # Let's just recalculate reps at the end of repair loop if we added many?
            # Or just assume it's fixed?
            # To be safe, let's recalculate reps after each addition?
            # No, too slow.
            # Let's just add the element and trust that it helps.
            # We will check validity at the end.

        # Check if repaired
        final_reps = calc_reps(repair_mask)
        if not np.any(final_reps == 0):
            best_mask = repair_mask
            best_A = mask_to_list(best_mask)
        else:
            # If still invalid, fallback to dense set
            mask = np.zeros(129, dtype=np.int32)
            for i in range(64, 129):
                mask[i] = 1
            return mask_to_list(mask)

    return best_A
