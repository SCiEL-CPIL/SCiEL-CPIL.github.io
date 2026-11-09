import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin before the 90s kill
    
    # Fallback valid set
    best_set = [31, 32, 34]
    best_size = 3
    
    MAX_SUM = 8000
    
    # Precompute divisibility checks for speed? 
    # Actually, checking `if (m*u) in sums_set` is O(1).
    # We can optimize by precomputing a boolean array for "is_divisible_pair" but sums are dynamic.
    
    def is_valid_full(S):
        """Check if set S is valid using full subset sum enumeration."""
        if not S:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        
        non_empty_set = sums - {0}
        for u in non_empty_set:
            # Check if any multiple of u is in the set
            # Only need to check multiples > u
            for m in range(2, MAX_SUM // u + 1):
                if (m * u) in non_empty_set:
                    return False
        return True

    def try_add(S, x, sums_set):
        """
        Try to add x to S. S is assumed valid.
        sums_set is the set of all subset sums of S (including 0).
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        new_sums = set()
        for s in sums_set:
            ns = s + x
            if ns in sums_set:
                return (False, None)
            new_sums.add(ns)
        
        all_sums = sums_set | new_sums
        non_empty_all_set = all_sums - {0}
        
        # Check divisibility among all non-empty sums
        # Optimization: We only need to check if new sums create divisibility with old sums or other new sums.
        # However, since we check all pairs, it's safer to just check the whole set.
        # But we can be smarter: 
        # 1. If u is in old_sums, and m*u is in new_sums -> invalid
        # 2. If u is in new_sums, and m*u is in old_sums -> invalid
        # 3. If u is in new_sums, and m*u is in new_sums -> invalid
        # This is effectively checking all pairs anyway.
        
        # To speed up:
        # Iterate through the smaller set of candidates for 'u'?
        # Or just iterate through all and check.
        
        # Let's try a slightly faster check:
        # For each u in non_empty_all_set:
        #   Check if any multiple is in non_empty_all_set
        
        # We can precompute a set of all values to check against.
        # The loop `for m in range(2, MAX_SUM // u + 1)` can be expensive if u is small.
        # But u >= 1. If u=1, we check up to 8000.
        # If we have many small sums, this is slow.
        
        # Alternative:
        # Build a list of sorted non-empty sums.
        # For each u, binary search for multiples? No, direct lookup in set is O(1).
        # The bottleneck is the range of m.
        
        # Let's stick to the simple set lookup but optimize the loop.
        for u in non_empty_all_set:
            # We only care if a multiple exists.
            # If u > MAX_SUM // 2, no multiple can exist (except 2u > MAX_SUM).
            # So we only iterate u up to MAX_SUM // 2.
            if u > MAX_SUM // 2:
                continue
            for m in range(2, MAX_SUM // u + 1):
                if (m * u) in non_empty_all_set:
                    return (False, None)
        
        return (True, all_sums)

    def greedy_build(candidates, start_S=None, start_sums=None):
        """Greedy build: iterate through candidates, add if valid."""
        if start_S is None:
            S = []
            sums_set = {0}
        else:
            S = list(start_S)
            sums_set = set(start_sums)
        
        for x in candidates:
            if x in S:
                continue
            ok, new_sums = try_add(S, x, sums_set)
            if ok:
                S.append(x)
                sums_set = new_sums
        
        return S, sums_set

    # Strategy: Randomized greedy with various sorting orders
    # Larger numbers often help avoid divisibility by smaller sums?
    # Or smaller numbers to keep sums small?
    # Let's try a mix.
    
    max_restarts = 5000  # Reduced to allow time for local search
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Create candidate list
        candidates = list(range(1, 257))
        
        # Randomly choose a strategy for ordering
        strat = random.randint(0, 2)
        if strat == 0:
            random.shuffle(candidates)
        elif strat == 1:
            # Descending
            candidates.sort(reverse=True)
            # Add noise
            for i in range(len(candidates)):
                if random.random() < 0.1:
                    j = random.randint(0, len(candidates)-1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
        else:
            # Ascending
            candidates.sort()
            for i in range(len(candidates)):
                if random.random() < 0.1:
                    j = random.randint(0, len(candidates)-1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]

        S, sums_set = greedy_build(candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Try to extend
            remaining = [x for x in range(1, 257) if x not in S]
            random.shuffle(remaining)
            S2, sums_set2 = greedy_build(remaining, S, sums_set)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2[:]

    # Local search: swap one element for another
    if best_size < 30 and time.time() - start_time < time_limit - 5:
        current_S = list(best_set)
        sums_set = {0}
        for x in current_S:
            new_sums = set()
            for s in sums_set:
                new_sums.add(s + x)
            sums_set = sums_set | new_sums
        
        for i in range(len(current_S)):
            if time.time() - start_time > time_limit - 2:
                break
            
            removed = current_S[i]
            S_reduced = current_S[:i] + current_S[i+1:]
            
            # Recompute sums without removed
            sums_reduced = {0}
            for x in S_reduced:
                new_sums = set()
                for s in sums_reduced:
                    new_sums.add(s + x)
                sums_reduced = sums_reduced | new_sums
            
            # Try adding each candidate
            for x in range(1, 257):
                if x in S_reduced:
                    continue
                ok, new_sums = try_add(S_reduced, x, sums_reduced)
                if ok:
                    new_S = S_reduced + [x]
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = new_S[:]

    # Final verification
    if len(best_set) > 0:
        if not is_valid_full(best_set):
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
