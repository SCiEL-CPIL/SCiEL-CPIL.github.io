import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    best_set = []
    
    def compute_all_sums(S):
        """Compute all triple sums for a set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(S[i] + S[j] + S[k])
        return sums

    def is_valid_addition(S, x, sums_set):
        # S is the current list, x is candidate
        # sums_set is the set of all triple sums of S
        # We need to check if adding x creates duplicate sums
        
        new_sums = set()
        
        # 1. x, x, x
        s = x + x + x
        if s in sums_set or s in new_sums:
            return False, None
        new_sums.add(s)
        
        # 2. x, x, y for y in S
        for y in S:
            s = x + x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # 3. x, y, z for y, z in S, y <= z
        # Note: y and z can be same
        # To avoid duplicates in generation, we iterate indices
        k = len(S)
        for i in range(k):
            y = S[i]
            for j in range(i, k):
                z = S[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    def run_greedy(order):
        S = []
        sums_set = set()
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = is_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S

    def local_search(S):
        """Try to improve set S by swapping elements."""
        if not S:
            return S
        if time.time() > deadline:
            return S
            
        S = sorted(S)
        sums_set = compute_all_sums(S)
        improved = True
        max_iterations = 200
        
        while improved and time.time() < deadline:
            improved = False
            for _ in range(max_iterations):
                if time.time() > deadline:
                    break
                    
                # Try to remove a random element and add a different one
                if len(S) < 2:
                    break
                    
                idx_to_remove = random.randint(0, len(S) - 1)
                old_val = S[idx_to_remove]
                
                # Try candidate values: nearby values, or random
                candidates = set()
                # Nearby values
                for delta in range(-50, 51):
                    v = old_val + delta
                    if 1 <= v <= 4096 and v not in S:
                        candidates.add(v)
                # Some random candidates
                for _ in range(20):
                    v = random.randint(1, 4096)
                    if v not in S:
                        candidates.add(v)
                        
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                        
                    # Check if we can replace old_val with new_val
                    # Remove old_val from sums
                    new_S = S[:idx_to_remove] + S[idx_to_remove+1:]
                    
                    # Compute sums without old_val
                    new_sums = set()
                    n = len(new_S)
                    for i in range(n):
                        for j in range(i, n):
                            for k in range(j, n):
                                new_sums.add(new_S[i] + new_S[j] + new_S[k])
                    
                    # Now check if adding new_val is valid
                    valid, added_sums = is_valid_addition(new_S, new_val, new_sums)
                    if valid:
                        # Accept the swap
                        S = new_S + [new_val]
                        S.sort()
                        sums_set = new_sums | added_sums
                        improved = True
                        break
                        
        return S

    # 1. Standard Greedy (Increasing)
    order_inc = range(1, 4097)
    S_inc = run_greedy(order_inc)
    if len(S_inc) > len(best_set):
        best_set = S_inc
        
    # 2. Standard Greedy (Decreasing)
    order_dec = range(4096, 0, -1)
    S_dec = run_greedy(order_dec)
    if len(S_dec) > len(best_set):
        best_set = S_dec

    # 3. Randomized Greedy with diverse strategies
    while time.time() < deadline:
        # Choose a random strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Random permutation
            order = list(range(1, 4097))
            random.shuffle(order)
        elif strategy < 0.5:
            # Biased toward smaller numbers (exponential distribution)
            order = [random.randint(1, 4096) for _ in range(4096)]
            order.sort()
            # Deduplicate while preserving some randomness
            seen = set()
            unique_order = []
            for x in order:
                if x not in seen:
                    seen.add(x)
                    unique_order.append(x)
            order = unique_order
        elif strategy < 0.7:
            # Start from a random point, go both directions
            start = random.randint(1, 4096)
            order = []
            for d in range(4096):
                left = start - d
                right = start + d
                if 1 <= left <= 4096:
                    order.append(left)
                if 1 <= right <= 4096 and right != left:
                    order.append(right)
            random.shuffle(order)
        elif strategy < 0.85:
            # Two-phase: first half random small, second half random large
            small = list(range(1, 2049))
            large = list(range(2049, 4097))
            random.shuffle(small)
            random.shuffle(large)
            order = small + large
        else:
            # Pure random with replacement then dedup
            order = [random.randint(1, 4096) for _ in range(4096)]
            seen = set()
            unique_order = []
            for x in order:
                if x not in seen:
                    seen.add(x)
                    unique_order.append(x)
            order = unique_order
            
        S_rand = run_greedy(order)
        
        # Apply local search if we have time
        if len(S_rand) > len(best_set) and time.time() < deadline - 10:
            S_improved = local_search(S_rand)
            if len(S_improved) > len(best_set):
                best_set = S_improved
            elif len(S_rand) > len(best_set):
                best_set = S_rand
        elif len(S_rand) > len(best_set):
            best_set = S_rand
            
    # Final local search on best set if we have time
    if best_set and time.time() < deadline - 5:
        S_final = local_search(best_set)
        if len(S_final) >= len(best_set):
            best_set = S_final
            
    return best_set
