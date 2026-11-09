import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN_SIZE = MAX_VAL + 1
    
    # Precompute indices for even sums to adjust for unordered pairs
    # s is in [128, 256]
    # s/2 is in [64, 128]
    # We need to know if s/2 is in A.
    
    def calc_reps(A_mask):
        """
        A_mask: numpy boolean array of length 129
        Returns: numpy array of counts for sums 128..256
        """
        # f is the indicator vector
        f = A_mask.astype(np.float32)
        
        # Convolution gives counts for ordered pairs
        # Length of f is 129. Convolution length is 257.
        # We need indices 128 to 256 (inclusive)
        conv = np.convolve(f, f, mode='full')
        
        # Extract the relevant slice
        # conv[k] is the sum for k.
        # We want sums s where 128 <= s <= 256.
        # So we take conv[128:257]
        ordered_counts = conv[128:257]
        
        # Convert to unordered counts
        # C[s] = (K[s] + I(s/2 in A)) / 2
        # s = 128 + i
        # s/2 = 64 + i/2
        # We need to check if 64 + i/2 is an integer and in A.
        
        counts = ordered_counts / 2.0
        
        # Add 0.5 for even sums where s/2 is in A
        # s is even if i is even (since 128 is even)
        # i = 0, 2, 4, ...
        # s/2 = 64 + i/2
        # idx_in_A = 64 + i/2
        
        # Vectorized update
        even_indices = np.arange(0, RANGE_SIZE, 2)
        if len(even_indices) > 0:
            # Calculate the index in A for each even sum
            # s = 128 + i
            # s/2 = 64 + i/2
            a_indices = 64 + even_indices // 2
            # Check if these indices are within bounds and in A
            # a_indices will be in [64, 128], which is valid for A_mask
            in_A = A_mask[a_indices]
            # Add 0.5 to counts at positions even_indices where in_A is True
            counts[even_indices[in_A]] += 0.5
            
        # Ensure integer output (counts should be integers)
        # Due to floating point, round to nearest int
        return np.round(counts).astype(np.int32)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    best_A_mask = None
    best_peak = float('inf')
    
    def update_best(A_mask, reps):
        nonlocal best_A_mask, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A_mask = A_mask.copy()

    # Phase 1: Greedy Peeling from Full Set
    # Start with A = [0, 128]
    A_mask = np.ones(DOMAIN_SIZE, dtype=bool)
    reps = calc_reps(A_mask)
    
    if not is_covered(reps):
        # This shouldn't happen with full set
        pass
        
    update_best(A_mask, reps)
    
    # Greedy Removal
    # We want to remove elements that reduce the peak or allow further reduction.
    # Strategy: 
    # 1. Find current peak.
    # 2. For each element x in A, simulate removal.
    # 3. If removal keeps coverage and reduces peak (or keeps peak same and reduces size), do it.
    # To be efficient, we prioritize elements that are "central" or part of peak sums.
    # But with fast calc_reps, we can just try all removals in each step.
    # |A| starts at 129. Each step tries 129 removals. 129 * 129 * cost_calc.
    # cost_calc is ~16k ops (conv). 129*129*16k ~ 260M ops. Might be too slow in Python loop overhead?
    # Let's optimize: only try removing elements that are part of the current max-sum pairs?
    # Or just limit the number of steps.
    
    # Better Greedy:
    # Iterate until no improvement.
    # In each iteration, find the element whose removal yields the best "score".
    # Score = (new_peak, new_size) lexicographically, subject to validity.
    
    current_peak = get_peak(reps)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_candidate_mask = None
        best_candidate_reps = None
        best_candidate_peak = current_peak
        best_candidate_size = len(np.where(A_mask)[0])
        
        # Get list of active elements
        active_elements = np.where(A_mask)[0]
        
        # To save time, if active_elements is large, we might sample? 
        # No, 129 is small. Let's just check all.
        for x in active_elements:
            new_mask = A_mask.copy()
            new_mask[x] = False
            
            new_reps = calc_reps(new_mask)
            
            if not is_covered(new_reps):
                continue
                
            new_peak = get_peak(new_reps)
            new_size = len(np.where(new_mask)[0])
            
            # Accept if peak is lower, or peak is same and size is smaller
            if new_peak < best_candidate_peak:
                best_candidate_peak = new_peak
                best_candidate_mask = new_mask
                best_candidate_reps = new_reps
                best_candidate_size = new_size
                improved = True
            elif new_peak == best_candidate_peak and new_size < best_candidate_size:
                best_candidate_mask = new_mask
                best_candidate_reps = new_reps
                best_candidate_size = new_size
                improved = True
                
        if improved:
            A_mask = best_candidate_mask
            reps = best_candidate_reps
            current_peak = best_candidate_peak
            update_best(A_mask, reps)
            
    # Phase 2: Local Search (Swaps)
    # Start from the best found so far
    A_curr = best_A_mask.copy()
    reps_curr = calc_reps(A_curr)
    peak_curr = get_peak(reps_curr)
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    stuck_counter = 0
    max_stuck = 100
    
    active_elements = np.where(A_curr)[0]
    inactive_elements = np.where(~A_curr)[0]
    
    while time.time() < deadline - 2 and T > min_T:
        if stuck_counter > max_stuck:
            break
            
        # Generate a swap move
        if len(active_elements) == 0 or len(inactive_elements) == 0:
            break
            
        x = random.choice(active_elements)
        y = random.choice(inactive_elements)
        
        new_mask = A_curr.copy()
        new_mask[x] = False
        new_mask[y] = True
        
        new_reps = calc_reps(new_mask)
        
        if not is_covered(new_reps):
            stuck_counter += 1
            continue
            
        new_peak = get_peak(new_reps)
        
        if new_peak < peak_curr:
            A_curr = new_mask
            reps_curr = new_reps
            peak_curr = new_peak
            active_elements = np.where(A_curr)[0]
            inactive_elements = np.where(~A_curr)[0]
            update_best(A_curr, reps_curr)
            stuck_counter = 0
        else:
            # Accept worse moves with probability
            delta = new_peak - peak_curr
            if delta == 0:
                # Neutral move, accept to shuffle
                if random.random() < 0.5:
                    A_curr = new_mask
                    reps_curr = new_reps
                    active_elements = np.where(A_curr)[0]
                    inactive_elements = np.where(~A_curr)[0]
                    stuck_counter = 0
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    A_curr = new_mask
                    reps_curr = new_reps
                    peak_curr = new_peak
                    active_elements = np.where(A_curr)[0]
                    inactive_elements = np.where(~A_curr)[0]
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
        T *= cooling
        
    # Final Polish: Try removing any remaining redundant elements
    if best_A_mask is not None:
        A = best_A_mask.copy()
        reps = calc_reps(A)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            active = np.where(A)[0]
            for x in active:
                new_A = A.copy()
                new_A[x] = False
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    old_peak = get_peak(reps)
                    if new_peak <= old_peak:
                        A = new_A
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break # Restart loop

    if best_A_mask is None:
        # Fallback: return a known valid set, e.g., [0, 128]
        fallback = np.ones(DOMAIN_SIZE, dtype=bool)
        return list(np.where(fallback)[0])
        
    return list(np.where(best_A_mask)[0])

