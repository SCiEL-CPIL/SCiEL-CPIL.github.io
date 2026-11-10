import time
from math import gcd
from functools import reduce
import random

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # Integer numerator for each reciprocal

    # We will search for the maximum size subset.
    # We use Branch and Bound.
    
    best_set = []
    best_len = 0
    
    def _update_best(current_set):
        nonlocal best_set, best_len
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    # Helper to check if adding n to sums is valid
    # Returns new sums set if valid, else None
    def _try_add(sums, n):
        # Check collision: (s + n) in sums for any s in sums
        # Since sums is a set, this is O(|sums|)
        # Optimization: if n is large, collisions are less likely?
        # We can just iterate.
        for s in sums:
            if s + n in sums:
                return None
        # If no collision, compute new sums
        # new_sums = sums | {s + n for s in sums}
        # To save time, we only construct if we actually recurse
        return sums

    # DFS with pruning
    # order: list of indices into universe
    def _dfs(pos, path, sums, order, nums):
        nonlocal best_len, best_set
        
        # Time check
        if time.time() > deadline:
            return

        # Prune
        if len(path) + (N - pos) <= best_len:
            return

        if pos == N:
            if len(path) > best_len:
                best_len = len(path)
                best_set = path[:]
            return

        idx = order[pos]
        n = nums[idx]
        v = universe[idx]

        # Try Include
        # Check collision
        valid = True
        for s in sums:
            if s + n in sums:
                valid = False
                break
        
        if valid:
            # Create new sums
            # We can optimize by only creating the set if we are going deep
            # But we need it for recursion.
            new_sums = sums.copy()
            for s in sums:
                new_sums.add(s + n)
            
            path.append(v)
            _dfs(pos + 1, path, new_sums, order, nums)
            path.pop()
            
            # If we found a solution of size N, we can stop early? 
            # No, we want max size. If we find N, we can break.
            if best_len == N:
                return

        # Try Exclude
        _dfs(pos + 1, path, sums, order, nums)

    # Multiple orderings to ensure we explore different parts of the tree
    # 1. Ascending Numerator (Descending Element: 22, 21, ..., 1)
    #    Small steps first.
    order_asc_num = list(range(N - 1, -1, -1))
    
    # 2. Descending Numerator (Ascending Element: 1, 2, ..., 22)
    #    Large steps first.
    order_desc_num = list(range(N))
    
    # 3. Randomized ordering (fixed seed for reproducibility)
    random.seed(42)
    order_rand = list(range(N))
    random.shuffle(order_rand)
    
    # Run DFS for each ordering
    # We share the best_len across runs to prune more effectively
    
    # Run 1: Ascending Numerator
    _dfs(0, [], {0}, order_asc_num, num)
    
    if time.time() > deadline:
        if best_set:
            return sorted(best_set)
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    # Run 2: Descending Numerator
    _dfs(0, [], {0}, order_desc_num, num)
    
    if time.time() > deadline:
        if best_set:
            return sorted(best_set)
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    # Run 3: Random
    _dfs(0, [], {0}, order_rand, num)

    # Local Search Polish
    # Try to add any remaining element to the best set found
    if best_set:
        best_set_set = set(best_set)
        current_sums = {0}
        # Rebuild sums for best_set
        for v in best_set:
            n = L // v
            new_sums = current_sums.copy()
            for s in current_sums:
                new_sums.add(s + n)
            current_sums = new_sums
            
        remaining = [v for v in universe if v not in best_set_set]
        # Try adding each remaining element
        for v in remaining:
            if time.time() > deadline:
                break
            n = L // v
            # Check if v can be added to current_sums
            valid = True
            for s in current_sums:
                if s + n in current_sums:
                    valid = False
                    break
            if valid:
                # Add it
                best_set.append(v)
                best_set_set.add(v)
                new_sums = current_sums.copy()
                for s in current_sums:
                    new_sums.add(s + n)
                current_sums = new_sums
                # Restart the loop to see if we can add more
                remaining = [x for x in remaining if x != v]
                # Actually, just break and restart the whole local search block?
                # Or just continue checking other remaining elements against the NEW set.
                # Since we modified current_sums, we should check the other remaining elements again.
                # But we are iterating over a fixed list. 
                # Let's just do a simple loop that keeps trying until no more can be added.
                break 

        # Repeat local search until no improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_set_set = set(best_set)
            current_sums = {0}
            for v in best_set:
                n = L // v
                new_sums = current_sums.copy()
                for s in current_sums:
                    new_sums.add(s + n)
                current_sums = new_sums
                
            remaining = [v for v in universe if v not in best_set_set]
            for v in remaining:
                n = L // v
                valid = True
                for s in current_sums:
                    if s + n in current_sums:
                        valid = False
                        break
                if valid:
                    best_set.append(v)
                    improved = True
                    break # Restart while loop to rebuild sums

    if best_set:
        return sorted(best_set)
    
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
