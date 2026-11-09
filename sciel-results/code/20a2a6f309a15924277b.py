import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Precompute divisibility relationships is not strictly necessary if we check on the fly,
    # but having a set of all possible sums is good.
    # Max sum for 30 numbers from 1..256 is ~7680.
    MAX_SUM = 8000

    def is_valid_full(S):
        """Check if set S is valid using full subset sum enumeration."""
        if not S:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            if sums & new_sums:
                return False
            sums = sums | new_sums
        
        non_empty = [s for s in sums if s != 0]
        non_empty_set = set(non_empty)
        # Sort to potentially break early? No, just check.
        for u in non_empty:
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
        non_empty_all_set = {s for s in all_sums if s != 0}
        
        # Check divisibility among all non-empty sums
        # Optimization: If we just added x, we only need to check:
        # 1. New sums vs Old sums
        # 2. New sums vs New sums
        # But checking all is simpler and $O(N \log N)$ where N is num sums.
        # For small k, this is fine.
        for u in non_empty_all_set:
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
    
    max_restarts = 10000
    
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
