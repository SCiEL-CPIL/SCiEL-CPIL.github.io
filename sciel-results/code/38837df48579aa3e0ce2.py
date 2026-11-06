def construct():
    import time
    import random
    from collections import defaultdict

    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
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
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit=40.0):
        n = len(vals)
        vals_set = set(vals)
        start_time = time.time()
        
        # 1. Initial Candidates
        candidates = []
        
        # Odds
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        # Positives
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        # Negatives
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        # Top half by value
        sorted_vals = sorted(vals, reverse=True)
        top_half = sorted_vals[:n//2 + 1]
        candidates.append(set(top_half))
        
        # Bottom half by value
        bottom_half = sorted_vals[-(n//2 + 1):]
        candidates.append(set(bottom_half))
        
        # Top half by absolute value
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        top_abs = sorted_abs[:n//2 + 1]
        candidates.append(set(top_abs))
        
        # Bottom half by absolute value
        bottom_abs = sorted_abs[-(n//2 + 1):]
        candidates.append(set(bottom_abs))
        
        # Random samples
        rng = random.Random(42)
        for _ in range(20):
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
        
        # Greedy constructions with different orderings
        def greedy_add(S, x):
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
            
            # For y in S, check if x-y in S (which implies y + (x-y) = x)
            for y in S:
                if x - y in S:
                    return False
            
            return True

        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
            sorted(vals, key=lambda x: random.Random(hash(x)).random()),
        ]
        
        for ordering in orderings:
            S = set()
            for x in ordering:
                if greedy_add(S, x):
                    S.add(x)
            candidates.append(S)
        
        # 2. Simulated Annealing / Local Search
        # We will try to improve each candidate.
        # Strategy: 
        # - Start with a candidate.
        # - Loop:
        #   - Try to add a random element. If it fails, identify conflicts.
        #   - Try to remove a random element from the conflict set.
        #   - Or, simpler: Randomly pick an element not in S, try to add. If fail, randomly pick an element from S to remove, and try to add again?
        #   - Even simpler and effective: "Add if possible" fixed point, then "Swap" moves.
        
        def try_improve(S, vals_set, max_iter=1000):
            S = set(S)
            improved = True
            iter_count = 0
            local_rng = random.Random(hash(frozenset(S)))
            
            # First, greedily add any possible elements until fixed point
            while improved and time.time() - start_time < time_limit:
                improved = False
                for x in vals_set:
                    if x in S:
                        continue
                    if greedy_add(S, x):
                        S.add(x)
                        improved = True
                        break # Restart loop with new S
                
                if not improved:
                    break
            
            # Now, try swaps: remove y, add x.
            # This is useful if x conflicts with y, but removing y allows adding x AND other things?
            # Or just if x is "better" than y? No, we want max size.
            # Swap is useful if removing y allows adding x, and x doesn't conflict with others, 
            # and maybe x allows adding z later?
            # Actually, a simpler move: 
            # Pick x not in S. Find all y in S such that {x, y} is not sum-free together (i.e., adding x requires removing y).
            # If we remove y, can we add x? Yes.
            # Does this help? Only if it unlocks other additions.
            
            # Let's try a random walk:
            for _ in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # Randomly pick an element to remove
                if not S:
                    break
                y = local_rng.choice(list(S))
                S.remove(y)
                
                # Try to add elements greedily
                added_any = False
                for x in vals_set:
                    if x in S:
                        continue
                    if greedy_add(S, x):
                        S.add(x)
                        added_any = True
                
                # If we didn't add anything, maybe we lost size.
                # If we added, size might have increased or stayed same.
                # We should keep the best S found.
            
            return S

        best_S = set()
        best_size = -1
        
        # Sort candidates by size, largest first, to prioritize good starting points
        candidates.sort(key=len, reverse=True)
        
        for S in candidates:
            if time.time() - start_time > time_limit:
                break
            
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
        
        # Allocate time based on instance size
        # n=60: 20s, n=120: 40s, n=240: 60s? Total 120s.
        # Let's use a proportional allocation.
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 60.0
            
        S = solve_instance(vals, t_lim)
        
        # Final verification
        if not is_sum_free(S):
            # Fallback to odds
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            # If we ran out of time, return what we have
            break

    # Ensure we return 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]
