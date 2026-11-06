def construct():
    import time
    from random import Random
    from itertools import combinations

    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S:
                    return False
        return True

    def solve_instance(vals):
        n = len(vals)
        vals_set = set(vals)
        
        # Generate candidate sets
        candidates = []
        
        # 1. Odds
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        # 2. Positives
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        # 3. Negatives
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        # 4. Top half by value (largest)
        sorted_vals = sorted(vals, reverse=True)
        top_half = sorted_vals[:n//2 + 1]
        candidates.append(set(top_half))
        
        # 5. Bottom half by value (smallest)
        bottom_half = sorted_vals[-(n//2 + 1):]
        candidates.append(set(bottom_half))
        
        # 6. Top half by absolute value
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        top_abs = sorted_abs[:n//2 + 1]
        candidates.append(set(top_abs))
        
        # 7. Bottom half by absolute value
        bottom_abs = sorted_abs[-(n//2 + 1):]
        candidates.append(set(bottom_abs))
        
        # 8. Random samples
        rng = Random(42)
        for _ in range(10):
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
        
        # Also try greedy construction from scratch
        # Greedy: start empty, add element if it doesn't create a sum-free violation
        # But this is O(n^3) if done naively. 
        # A better greedy: sort by some heuristic, add if possible.
        
        def greedy_add(S, x, vals_set):
            # Check if adding x to S violates sum-free property
            # Conflicts:
            # 1. x + x = 2x in S
            # 2. x + y = z for y, z in S
            # 3. y + z = x for y, z in S
            
            if 2*x in S:
                return False
            
            # For y in S, check if x+y in S
            for y in S:
                if x + y in S:
                    return False
            
            # For y, z in S, check if y+z = x
            # This is equivalent to: for y in S, if x-y in S and x-y != y?
            # Actually, if y+z=x, then z=x-y.
            # So for y in S, if x-y in S, then we have a conflict?
            # Wait, if y and z are in S and y+z=x, then (y,z,x) is a bad triple.
            # So we need to check if there exist y, z in S such that y+z=x.
            # This is the same as: for y in S, if x-y in S.
            # But we must be careful about y=z.
            # If y=z, then 2y=x. If x-y = y, then 2y=x.
            # So checking `if x-y in S` covers both cases.
            
            for y in S:
                if x - y in S:
                    return False
            
            return True

        # Try greedy with different orderings
        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
        ]
        
        for ordering in orderings:
            S = set()
            for x in ordering:
                if greedy_add(S, x, vals_set):
                    S.add(x)
            candidates.append(S)
        
        # Local search / Hill Climbing on the best candidates
        # We will try to improve each candidate by swapping elements or adding/removing.
        # Since we want to maximize size, we mostly want to add.
        # But adding might require removing others.
        
        def try_improve(S, vals_set, max_iter=500):
            S = set(S)
            improved = True
            iter_count = 0
            while improved and iter_count < max_iter and time.time() < deadline:
                improved = False
                iter_count += 1
                # Try adding each element not in S
                for x in vals_set:
                    if x in S:
                        continue
                    # Check if we can add x
                    # This requires removing some elements from S.
                    # Let's find the minimal set of elements to remove to allow adding x.
                    # This is the same logic as the previous program's `solve_instance` inner loop.
                    # But doing this for every x in every iteration is expensive.
                    
                    # Let's stick to simple moves:
                    # 1. Add x if possible (no removals needed)
                    if greedy_add(S, x, vals_set):
                        S.add(x)
                        improved = True
                        # Break or continue? If we add, the set changes.
                        # Let's break to restart the loop with the new S.
                        break
                    
                    # 2. Swap x with y: remove y, add x.
                    # This is only useful if it allows adding more later? 
                    # Or if it increases size? No, size stays same.
                    # Unless adding x allows adding others that y didn't.
                    # This is complex. Let's skip swaps for now.
                    
                    # 3. Remove y to allow adding x?
                    # If removing y allows adding x, and net gain is +1 (remove 1, add 1 => 0 gain).
                    # But maybe removing y allows adding x AND z?
                    # This is getting complicated.
                    
                    # Let's try a different local search:
                    # Randomly remove an element, then try to add others greedily?
                    # Or: try to add x by removing a minimal conflict set.
                
                # Alternative Local Search:
                # For each x not in S, find minimal removals R such that (S \ R) U {x} is sum-free.
                # If len(R) < 1, we already handled it (greedy_add).
                # If len(R) == 1, we swap.
                # If len(R) > 1, we might still do it if it unlocks future gains?
                # For now, let's just do the "add if possible" pass repeatedly until no more additions.
                # This is a fixed point of the "add if possible" operator.
                
                # Wait, the loop above breaks on first add. 
                # So it will add one, restart, add another, etc.
                # This is a greedy algorithm that repeatedly adds any possible element.
                # The order of checking x matters.
                # In the current loop, we iterate through `vals_set` (a set, so arbitrary order in Python <3.7, but stable in 3.7+ for insertion? No, sets are unordered).
                # To make it deterministic, we should iterate over a sorted list.
                
            return S

        # Apply improvement to all candidates
        best_S = set()
        best_size = -1
        
        for S in candidates:
            # Verify sum-free
            if not is_sum_free(S):
                continue
            
            S_improved = try_improve(S, vals_set)
            
            # Verify again
            if not is_sum_free(S_improved):
                continue
                
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        S = solve_instance(vals)
        
        # Final verification
        if not is_sum_free(S):
            # Fallback to odds
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            # If we ran out of time, return what we have
            # But we need to process all 3? 
            # The contract says "three subsets". 
            # If we time out, we should return best effort.
            # Since we process sequentially, if we time out on instance 1, we have no results for 2 and 3.
            # We should allocate time per instance.
            pass

    return results
