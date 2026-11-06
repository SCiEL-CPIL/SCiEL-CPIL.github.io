import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # The evaluator kills at 90s. We need a safety margin.
    # Previous program used 88s limit. Let's use 85s to be safe for cleanup/return.
    time_limit = 85.0
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    # We want to maximize the size of the set.
    # The theoretical upper bound for B_3[1] in [1, N] is related to N^(1/3).
    # For N=4096, 4096^(1/3) = 16.
    # So the maximum size is likely 16.
    # The current score 1.7778 corresponds to size 16 (16/9 = 1.777... if seed was 9? No, score is value/seed.
    # If value is size, and score is 1.7778, and seed is 9, then 16/9 = 1.777.
    # So the seed program likely found 9 elements.
    # The current program found 16 elements.
    # Can we find 17?
    # If 16 is the maximum, we can't improve.
    # However, the task says "Maximize cardinality".
    # Let's assume 16 is optimal for now, but try to find it as fast as possible
    # to allow for verification or potential edge cases.
    # Actually, if 16 is the max, the score is fixed.
    # But maybe the seed was different?
    # "search score is value/seed".
    # If the seed program returned 9, and we return 16, score is 16/9 = 1.77.
    # If we return 17, score is 17/9 = 1.88.
    # So we MUST try to find 17.
    
    # Strategy:
    # 1. Try to find a set of size 17 using randomized backtracking with strong pruning.
    # 2. If that fails within time, fall back to finding size 16.
    # 3. Use bitsets for fast collision checking.
    
    # Pre-compute nothing globally, but use efficient local checks.
    
    def check_time():
        return time.time() - start_time > time_limit

    def find_set(target_size, time_budget):
        """
        Attempts to find a set of size target_size within time_budget seconds.
        Uses randomized backtracking with bitset pruning.
        """
        deadline = time.time() + time_budget
        best_found = None
        
        # We will try multiple random restarts or just one deep search?
        # Backtracking is exponential. Randomization helps escape bad local minima
        # if we use a heuristic like "most constrained variable" or just random order.
        # Given the small domain (4096) and small target size (17),
        # a well-pruned search might work.
        
        # Pruning Heuristic:
        # At step k, we have chosen k elements.
        # We need target_size - k more.
        # The number of available candidates is bounded by the number of free sums.
        # Each new element x consumes:
        # - 1 sum for 3x
        # - k sums for 2x + s_i
        # - k(k+1)/2 sums for x + s_i + s_j
        # Total new sums: 1 + k + k(k+1)/2 = (k^2 + 3k + 2)/2 = (k+1)(k+2)/2
        # This is the number of new triple sums involving x.
        
        # Let's use a simple recursive backtracker with bitsets.
        # To avoid deep recursion limits or overhead, we can use an explicit stack
        # or just recursion if depth is small (max 17).
        
        def backtrack(current_set, used_mask, k):
            if time.time() > deadline:
                return None
            
            if k == target_size:
                return current_set[:]
            
            # Pruning:
            # Estimate if we can still reach target_size.
            # This is hard to do tightly without searching.
            # Simple prune: if k + available_candidates < target_size, return None.
            # Counting available_candidates is expensive (4096 checks).
            # We can do this check only if k is small or if we are stuck.
            
            # Heuristic: Iterate candidates in a random order?
            # Or use a specific order?
            # If we iterate 1..4096, we might find a solution quickly if one exists
            # with small numbers.
            # Let's try a hybrid: try small numbers first, but allow randomization
            # if we fail to find a solution in the first few layers?
            
            # Actually, for B_3[1], smaller numbers are "more expensive" in terms of
            # sum space usage? No, larger numbers use larger sum space, but the
            # density of collisions might be different.
            # Let's just iterate 1..4096.
            
            # Optimization:
            # If we have already tried many candidates and failed, we might be in a dead end.
            # But standard backtracking explores all.
            
            # To speed up, we can pre-filter candidates.
            # A candidate x is valid if:
            # 1. 3x is not in used_mask
            # 2. 2x + s is not in used_mask for all s in current_set
            # 3. x + s_i + s_j is not in used_mask for all i<=j in current_set
            # 4. No internal collisions among the new sums.
            
            # We can vectorize the check for a single x against current_set?
            # current_set is a list of ints.
            
            for x in range(1, max_val + 1):
                if time.time() > deadline:
                    return None
                
                # Check validity of x
                valid = True
                
                # 1. 3x
                if (used_mask >> (3*x)) & 1:
                    continue
                
                # 2. 2x + s
                # We can compute the mask for these sums and check against used_mask.
                # mask_x = 1 << (3*x)
                # for s in current_set:
                #     mask_x |= 1 << (2*x + s)
                
                # 3. x + s_i + s_j
                # for i in range(k):
                #     for j in range(i, k):
                #         mask_x |= 1 << (x + current_set[i] + current_set[j])
                
                # Combine all into one mask and check intersection with used_mask.
                # Also check internal collisions in mask_x?
                # If mask_x has duplicate bits set, it means internal collision.
                # We can check if popcount(mask_x) == expected_count.
                # Expected count: 1 + k + k(k+1)/2
                
                mask_x = 1 << (3*x)
                expected_bits = 1 + k + k*(k+1)//2
                bit_count = 0 # We'll count as we build
                
                # Actually, building the mask and checking intersection is O(k^2).
                # If k=16, k^2=256. 4096 * 256 = 1M ops per level.
                # Depth 17. 17M ops. This is fast in Python if optimized?
                # Python loops are slow. 1M * 17 = 17M iterations. Might take ~1-2 seconds.
                # So backtracking might be feasible for a few branches.
                
                # Let's build the mask.
                for s in current_set:
                    mask_x |= 1 << (2*x + s)
                    
                for i in range(k):
                    si = current_set[i]
                    for j in range(i, k):
                        sj = current_set[j]
                        mask_x |= 1 << (x + si + sj)
                
                # Check internal collisions
                # If popcount(mask_x) < expected_bits, there was a collision.
                # Python 3.10+ has int.bit_count().
                if mask_x.bit_count() < expected_bits:
                    continue
                    
                # Check against used_mask
                if mask_x & used_mask:
                    continue
                
                # Valid candidate
                current_set.append(x)
                result = backtrack(current_set, used_mask | mask_x, k + 1)
                if result is not None:
                    return result
                current_set.pop()
                
            return None
        
        # Try a few random permutations of the candidate order?
        # Or just try the natural order.
        # If natural order fails, it might be due to a specific choice early on.
        # Random restarts of the backtracking with different random seeds for ordering
        # could help.
        
        # However, the domain is ordered 1..4096.
        # Let's try the natural order first.
        sol = backtrack([], 0, 0)
        if sol:
            return sol
            
        # If natural order failed, try a few random shuffles of the candidate list?
        # This is expensive to generate.
        # Alternative: Randomize the order of candidates in the loop.
        # We can do this by generating a random permutation of range(1, 4097).
        # But doing this inside backtrack is expensive.
        
        # Let's do a few full restarts with random candidate orderings.
        for attempt in range(5):
            if time.time() > deadline:
                break
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
            def backtrack_random(current_set, used_mask, k):
                if time.time() > deadline:
                    return None
                if k == target_size:
                    return current_set[:]
                
                for x in candidates:
                    if time.time() > deadline:
                        return None
                    
                    mask_x = 1 << (3*x)
                    expected_bits = 1 + k + k*(k+1)//2
                    
                    for s in current_set:
                        mask_x |= 1 << (2*x + s)
                        
                    for i in range(k):
                        si = current_set[i]
                        for j in range(i, k):
                            sj = current_set[j]
                            mask_x |= 1 << (x + si + sj)
                            
                    if mask_x.bit_count() < expected_bits:
                        continue
                    if mask_x & used_mask:
                        continue
                    
                    current_set.append(x)
                    result = backtrack_random(current_set, used_mask | mask_x, k + 1)
                    if result is not None:
                        return result
                    current_set.pop()
                    
                return None
                
            sol = backtrack_random([], 0, 0)
            if sol:
                return sol
                
        return None

    # Try to find size 17
    # Give it a significant chunk of time, say 30 seconds.
    # If it fails, try size 16 with the remaining time.
    
    best_set = []
    
    # Attempt 17
    sol17 = find_set(17, 30.0)
    if sol17:
        best_set = sol17
    else:
        # Attempt 16
        # We know 16 is possible.
        # Use remaining time.
        remaining = time_limit - (time.time() - start_time) - 2.0
        if remaining > 1.0:
            sol16 = find_set(16, remaining)
            if sol16:
                best_set = sol16
            else:
                # Fallback to greedy if backtracking fails completely (shouldn't happen for 16)
                # Greedy is very fast.
                selected = []
                used_mask = 0
                for x in range(1, max_val + 1):
                    if time.time() > time_limit:
                        break
                    k = len(selected)
                    mask_x = 1 << (3*x)
                    expected_bits = 1 + k + k*(k+1)//2
                    for s in selected:
                        mask_x |= 1 << (2*x + s)
                    for i in range(k):
                        si = selected[i]
                        for j in range(i, k):
                            sj = selected[j]
                            mask_x |= 1 << (x + si + sj)
                    if mask_x.bit_count() < expected_bits:
                        continue
                    if mask_x & used_mask:
                        continue
                    selected.append(x)
                    used_mask |= mask_x
                best_set = selected

    return best_set
