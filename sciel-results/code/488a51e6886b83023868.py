"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute mapping from sum value to index in target_sums
    # sum -> index (0 to 128), or -1 if out of range
    sum_to_idx = np.full(max_sum + 1, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts(A_list):
        """Compute representation counts for target sums given list A."""
        counts = np.zeros(num_targets, dtype=np.int32)
        m = len(A_list)
        # Efficient double loop is hard to vectorize simply without massive memory,
        # but we can use numpy operations on the sorted array if needed.
        # For m ~ 40-60, O(m^2) is ~ 3600 ops. Very fast.
        for i in range(m):
            ai = A_list[i]
            # Pair (ai, ai)
            s = 2 * ai
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] += 1
            # Pairs (ai, aj) for j > i
            # We can vectorize the inner loop
            if i + 1 < m:
                aj_s = A_list[i+1:] + ai
                # Get indices
                idxs = sum_to_idx[aj_s]
                # Only count valid indices
                valid_mask = idxs >= 0
                if np.any(valid_mask):
                    valid_idxs = idxs[valid_mask]
                    # bincount to add 1 for each occurrence
                    # valid_idxs are in range [0, num_targets)
                    np.add.at(counts, valid_idxs, 1)
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def initial_greedy_solution():
        """
        Construct A using a greedy set cover approach that balances multiplicities.
        1. Start with empty A.
        2. While not all sums covered, find the sum with lowest count.
        3. Find element x in [0, 128] not in A that covers the most uncovered/low-count sums.
        4. Add x.
        """
        A = set()
        counts = np.zeros(num_targets, dtype=np.int32)
        
        # Precompute for each potential element x, which sums it contributes to
        # x contributes to sums: 2x, x+y for all y in A.
        # This is dynamic.
        
        # Heuristic: 
        # To cover [128, 256], we need elements.
        # Min element must be <= 128 (trivial).
        # Max element must be >= 128 (since max sum 256 = a+b, if a,b <= 128, max sum is 256 only if 128 in A).
        # Actually, if max(A) < 128, max sum < 256. So 128 MUST be in A?
        # If 128 not in A, max sum is at most 127+127=254. So 255, 256 uncovered.
        # Thus 128 is required? No, wait.
        # If A contains 128, sum 256 is covered by (128,128).
        # If A does not contain 128, max sum is 2*max(A). If max(A)=127, max sum 254.
        # So 128 MUST be in A to cover 256?
        # Yes, because sums are a+b with a,b in A. Max possible sum is 2*max(A).
        # To cover 256, we need 2*max(A) >= 256 => max(A) >= 128.
        # Since A subset [0, 128], max(A) must be 128.
        # So 128 is always in A.
        
        A.add(128)
        # Update counts for 128
        s = 256
        idx = sum_to_idx[s]
        if idx >= 0: counts[idx] += 1
        
        # We need to cover 128..255.
        # Let's greedily add elements that cover the most "deficit" sums.
        
        # To speed up, we can precompute the "coverage vector" for each x given current A.
        # But A changes.
        
        # Simple Greedy:
        # At each step, for each candidate x not in A, compute the new counts if x added.
        # Pick x that minimizes the new peak (or maximizes coverage of zeros).
        
        # This is O(129 * M * num_targets) per step? No, O(129 * M) to compute counts delta.
        # M is current size.
        
        # Let's try a simpler heuristic first:
        # Random initialization with 128, then greedy add.
        
        # Better: Start with a random valid set, then improve.
        # But the prompt asks for a DIFFERENT strategy. 
        # "Greedy Set Cover" was mentioned in inspiration 1 (score 5.5).
        # Current is SA (score 6.6).
        
        # Let's try a "Balanced Greedy Construction" followed by "Peak-Reduction Local Search".
        
        # 1. Construction:
        # We know 128 is in A.
        # To cover 128, we need a+b=128. 
        # To cover 129, a+b=129.
        
        # Let's use a greedy approach to build A such that counts are as flat as possible.
        # Start with A={128}.
        # While any count is 0:
        #   For each x in [0, 128] \ A:
        #     Calculate how many 0s it covers.
        #     Tie-break: how many 1s it covers? (to balance)
        #   Pick best x.
        
        # This might be slow if done naively.
        # Optimization:
        # Keep track of counts.
        # When considering adding x, the new counts are counts + delta(x).
        # delta(x) has 1s at indices corresponding to sums 2x and x+y for y in A.
        
        # Let's implement this.
        
        A = {128}
        counts = np.zeros(num_targets, dtype=np.int32)
        counts[sum_to_idx[256]] += 1
        
        # Precompute sums for each x with current A? No, A changes.
        
        while True:
            # Check if valid
            if np.all(counts > 0):
                break
            
            # Find best x to add
            best_x = -1
            best_score = -1
            best_new_counts = None
            
            # To save time, we can sample candidates if the set is large?
            # No, 129 candidates is small.
            
            # We want to minimize the new peak, or maximize the minimum count?
            # Objective: Minimize peak.
            # Greedy heuristic: Minimize the new max count?
            # Or: Maximize the number of sums that go from 0 to 1?
            
            # Let's try: Score = - (new_peak) + 0.1 * (number of sums covered that were 0)
            # Actually, just minimizing new peak is a good local heuristic.
            
            for x in range(129):
                if x in A:
                    continue
                
                # Calculate delta
                # Sums added: 2x, and x+y for y in A
                delta = np.zeros(num_targets, dtype=np.int32)
                
                s = 2*x
                idx = sum_to_idx[s]
                if idx >= 0:
                    delta[idx] += 1
                
                # x + y for y in A
                # A is a set. Convert to list for numpy op?
                # A_list = list(A)
                # sums = x + np.array(A_list)
                # idxs = sum_to_idx[sums]
                # valid = idxs >= 0
                # np.add.at(delta, idxs[valid], 1)
                
                # This is O(|A|) per x. Total O(129 * |A|).
                # |A| grows up to ~50. 129*50 = 6450 ops. Very fast.
                
                A_list = list(A)
                if A_list:
                    sums = x + np.array(A_list)
                    idxs = sum_to_idx[sums]
                    valid_mask = idxs >= 0
                    if np.any(valid_mask):
                        np.add.at(delta, idxs[valid_mask], 1)
                
                new_counts = counts + delta
                new_peak = np.max(new_counts)
                num_zeros_new = np.sum(new_counts == 0)
                
                # Score: We want low peak and low zeros.
                # If we have zeros, we MUST cover them.
                # So primary objective: reduce zeros.
                # Secondary: reduce peak.
                
                # If current has zeros, we prioritize x that covers most zeros.
                # If current has no zeros, we prioritize x that minimizes peak.
                
                if np.any(counts == 0):
                    # Count how many zeros are covered by x
                    # A zero is covered if delta[idx] > 0 and counts[idx] == 0
                    covered_zeros = np.sum((delta > 0) & (counts == 0))
                    # Also consider if it reduces peak?
                    # Just maximize covered_zeros. Tie break by lower peak.
                    score = covered_zeros * 100 - new_peak
                else:
                    # Already valid. We are in improvement phase?
                    # No, this loop is for construction. If valid, we break.
                    # So this case shouldn't happen if we break on valid.
                    score = -new_peak
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_counts = new_counts
            
            if best_x == -1:
                # No candidates? Should not happen.
                break
                
            # Add best_x
            A.add(best_x)
            counts = best_new_counts
            
            # Safety check for time
            if time.time() > deadline - 10.0:
                break
        
        return A, calculate_peak(counts)

    # Phase 1: Construct initial valid solution
    best_A, best_peak = initial_greedy_solution()
    current_A = set(best_A)
    current_counts = get_counts(list(current_A))
    current_peak = calculate_peak(current_counts)
    
    # If construction failed to make it valid, use a fallback dense set
    if current_peak >= 1000:
        # Fallback: Use a known valid structure
        # e.g., A = {0, 1, ..., 64} U {64, ..., 128}?
        # Just use random restarts of greedy
        for _ in range(10):
            A_try, peak_try = initial_greedy_solution()
            if peak_try < current_peak:
                current_A = A_try
                current_counts = get_counts(list(current_A))
                current_peak = peak_try
                best_A = set(current_A)
                best_peak = current_peak
            if time.time() > deadline - 5.0:
                break
        if current_peak >= 1000:
            # Absolute fallback
            return list(range(129))

    # Phase 2: Local Search to minimize peak
    # Moves: Swap x in A with y not in A.
    # Or: Remove x (if valid), Add x (if reduces peak? no, adding increases peak usually).
    # Swapping is the main move.
    
    T_start = 2.0
    T_end = 0.01
    alpha = 0.9995
    
    T = T_start
    steps = 0
    
    # Precompute list of all elements for sampling
    all_elements = list(range(129))
    
    while time.time() < deadline - 2.0 and T > T_end:
        steps += 1
        
        # Generate move: Swap
        # Pick x in A
        x = random.choice(list(current_A))
        # Pick y not in A
        # To speed up, we can maintain a list of not-in-A? 
        # Or just sample and check.
        # Sampling from 129 is fast.
        y = random.choice(all_elements)
        
        if x == y:
            continue
            
        if y in current_A:
            # If y is in A, this is not a swap with outside.
            # We want to swap x (in) with y (out).
            # If y is in A, we can't "add" it.
            # So we need y not in A.
            # Let's retry or pick differently.
            # For simplicity, if y in A, we skip this move (or swap x with another).
            # Better: explicitly pick y from complement.
            # Complement size ~ 129 - |A|.
            # We can construct complement list? O(129).
            # Or just sample until not in A.
            # Since |A| is small (~40), probability y in A is ~30%.
            # Expected 1.4 tries.
            while y in current_A:
                y = random.choice(all_elements)
        
        # Perform swap
        new_A = set(current_A)
        new_A.remove(x)
        new_A.add(y)
        
        # Calculate new counts
        # Optimization: We can update counts incrementally.
        # Remove contributions of x, add contributions of y.
        # But get_counts is fast enough for |A|~50.
        new_counts = get_counts(list(new_A))
        new_peak = calculate_peak(new_counts)
        
        delta_cost = new_peak - current_peak
        
        if delta_cost < 0:
            current_A = new_A
            current_counts = new_counts
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
        else:
            prob = np.exp(-delta_cost / T)
            if random.random() < prob:
                current_A = new_A
                current_counts = new_counts
                current_peak = new_peak
        
        T *= alpha
        
        # Periodic check
        if steps % 500 == 0 and time.time() > deadline - 2.0:
            break

    # Final verification
    if current_peak < best_peak:
        best_peak = current_peak
        best_A = set(current_A)
        
    final_counts = get_counts(list(best_A))
    if np.any(final_counts == 0):
        # Fallback
        return list(range(129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
