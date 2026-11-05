import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    best_set = []
    best_len = 0
    
    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        S = list(S)
        k = len(S)
        sums = set()
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def get_new_sums(x, S):
        """Get all new triple sums when adding x to set S."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y) # Wait, x+2y is same as 2y+x. 
            # My previous logic: 
            # 3x
            # 2x+y
            # x+2y
            # x+y+z
            # Let's be precise.
            # Triples involving x:
            # (x,x,x) -> 3x
            # (x,x,y) -> 2x+y (y in S)
            # (x,y,y) -> x+2y (y in S)
            # (x,y,z) -> x+y+z (y,z in S, y!=z or y==z handled by loops)
            # Actually, if we iterate i<=j in S for y,z, we cover y==z.
            # So:
            # 1. 3x
            # 2. 2x+y for y in S
            # 3. x+2y for y in S  (Note: if y=z, this is x+2y. If y!=z, it's part of x+y+z)
            # 4. x+y+z for y,z in S with y<=z.
            # Wait, (x,y,y) is covered by x+2y.
            # (x,y,z) with y<z is covered by x+y+z.
            # So we need:
            # - 3x
            # - 2x+y for all y in S
            # - x+2y for all y in S
            # - x+y+z for all y,z in S with y<z
            
        # Let's rewrite get_new_sums correctly and efficiently
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                new_sums.add(x + S[i] + S[j])
        return new_sums

    def can_add(x, S, existing_sums):
        """Check if x can be added to S without collision."""
        new_sums = get_new_sums(x, S)
        if new_sums & existing_sums:
            return False, None
        # Check internal uniqueness of new_sums
        # Expected count:
        # 1 (3x) + k (2x+y) + k (x+2y) + k(k-1)/2 (x+y+z, y<z)
        k = len(S)
        expected = 1 + k + k + k * (k - 1) // 2
        if len(new_sums) != expected:
            return False, None
        return True, new_sums

    def compute_all_sums(S):
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums

    # Strategy 1: Power of 3 base (quick baseline)
    power3_set = [3**i for i in range(8) if 3**i <= n]
    if is_valid_set(power3_set):
        best_set = list(power3_set)
        best_len = len(power3_set)

    # Strategy 2: Randomized Greedy with multiple seeds
    # We run many independent greedy searches.
    # To improve performance, we limit the number of candidates checked per run
    # if the set grows large, or just rely on the time check.
    
    num_seeds = 500
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > best_len:
            best_set = list(S)
            best_len = len(S)
            
            # If we find a significantly better set, we might want to try to improve it further
            # via local search (swaps) in a dedicated phase later.

    # Strategy 3: Local Search / Hill Climbing on the best set found
    # Try to increase the size by swapping elements.
    if best_len > 0:
        S = list(best_set)
        existing_sums = compute_all_sums(S)
        
        # Try to add elements first
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S:
                continue
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > best_len:
            best_set = list(S)
            best_len = len(S)

        # Now try swaps to break local minima
        # We try to remove one element and add a different one to allow future additions
        # This is a heuristic; we do a limited number of random swap attempts
        S = list(best_set)
        existing_sums = compute_all_sums(S)
        
        max_swap_attempts = 200
        for _ in range(max_swap_attempts):
            if time.time() > deadline:
                break
            
            if len(S) == 0:
                break
                
            # Pick a random element to remove
            idx_to_remove = random.randint(0, len(S) - 1)
            removed = S[idx_to_remove]
            S_copy = S.copy()
            S_copy.pop(idx_to_remove)
            
            # Recompute sums for S_copy
            sums_copy = compute_all_sums(S_copy)
            
            # Try to add a new element
            added = False
            # Try random candidates first, then sequential? 
            # Sequential is better for finding "small" additions, but random might find better structure.
            # Let's try a mix: a few random, then sequential from 1.
            
            # Try to add any valid element
            # To save time, we don't scan all 4096 every time.
            # We pick a random candidate.
            for _ in range(50):
                if time.time() > deadline:
                    break
                x = random.randint(1, n)
                if x in S_copy or x == removed:
                    continue
                valid, new_sums = can_add(x, S_copy, sums_copy)
                if valid:
                    S_copy.append(x)
                    sums_copy.update(new_sums)
                    added = True
                    break
            
            if added:
                # If we added an element, check if we can add more greedily
                # (This is optional, but might help)
                # Let's just accept the swap if it allows the set to remain valid and potentially grow later.
                # Actually, if we just swap, size remains same. We want to GROW.
                # So we should only accept the swap if it leads to a state where we can add MORE elements.
                
                # Try to add more elements to S_copy
                original_size = len(S_copy)
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if x in S_copy:
                        continue
                    valid, new_sums = can_add(x, S_copy, sums_copy)
                    if valid:
                        S_copy.append(x)
                        sums_copy.update(new_sums)
                
                if len(S_copy) > best_len:
                    best_set = list(S_copy)
                    best_len = len(S_copy)
                    S = list(S_copy)
                    existing_sums = sums_copy
                else:
                    # If we didn't grow, maybe this swap is still useful?
                    # No, we only care about maximizing size.
                    # So we revert.
                    pass

    # Final validation
    if best_set and not is_valid_set(best_set):
        # Fallback to power of 3
        best_set = [3**i for i in range(8) if 3**i <= n]
    
    return best_set
