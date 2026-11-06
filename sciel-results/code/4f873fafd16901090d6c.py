def construct():
    import time
    import random
    from collections import defaultdict

    deadline = time.time() + 145.0

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
        start_time = time.time()
        
        # Build conflict graph: node i conflicts with node j if they can't both be in a sum-free set
        # Two elements a, b conflict if:
        # 1. a + b is in vals (then a+b would be in S, violating sum-free)
        # 2. 2*a is in vals and a != b (but this is handled by 1 when a=b)
        # Actually, we need: a, b, c all in S with a+b=c is forbidden.
        # So for each pair (a,b) in S, a+b must NOT be in S.
        # This means: if a+b = c for some c in vals, then not all of {a,b,c} can be in S.
        # In the conflict graph: a conflicts with c if there exists b in vals such that a+b=c
        # and b would also need to be excluded... 
        # 
        # Better approach: The sum-free condition is: for all a,b,c in S, a+b != c.
        # This is a 3-uniform hypergraph independence problem.
        # But we can reduce to pairwise conflicts by noting:
        # If we include a and b, then a+b must NOT be in S.
        # So the conflict is: a and (a+b) conflict IF b is also in S.
        # This is not pairwise! It's a 3-way constraint.
        #
        # Alternative: Use a different formulation.
        # A set S is sum-free iff for every a in S, the set {a+b : b in S, b != -a} is disjoint from S.
        # 
        # Let me think differently. The standard approach for sum-free sets:
        # S is sum-free iff S ∩ (S+S) = ∅, where S+S = {a+b : a,b in S}.
        # 
        # For the MIS formulation, I'll use a different strategy:
        # Try multiple random orderings and greedy, plus local search with proper conflict detection.
        
        vals_set = set(vals)
        
        def can_add(S, x):
            """Check if x can be added to S while maintaining sum-free."""
            if 2*x in S:
                return False
            for y in S:
                if x + y in S:
                    return False
            for y in S:
                if x - y in S:
                    return False
            return True

        def greedy_order(ordering):
            S = set()
            for x in ordering:
                if can_add(S, x):
                    S.add(x)
            return S

        def local_search(S, max_iter=500):
            """Improve S by trying swaps and additions."""
            S = set(S)
            local_rng = random.Random(hash(frozenset(S)))
            best_S = set(S)
            best_size = len(S)
            
            for _ in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # Try to add a random element
                if random.random() < 0.5:
                    candidates_to_add = [x for x in vals if x not in S]
                    if not candidates_to_add:
                        continue
                    x = local_rng.choice(candidates_to_add)
                    if can_add(S, x):
                        S.add(x)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                else:
                    # Try a swap: remove y, add x
                    if not S:
                        continue
                    y = local_rng.choice(list(S))
                    S.remove(y)
                    # Try to add multiple elements
                    for x in vals:
                        if x in S:
                            continue
                        if can_add(S, x):
                            S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    else:
                        # Revert if it got worse
                        S.remove(x) if x in S else None
                        S.add(y)
            
            return best_S

        # Generate diverse starting points
        candidates = []
        
        # Structural candidates
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        sorted_vals = sorted(vals, reverse=True)
        candidates.append(set(sorted_vals[:n//2 + 1]))
        candidates.append(set(sorted_vals[-(n//2 + 1):]))
        
        sorted_abs = sorted(vals, key=lambda x: -abs(x))
        candidates.append(set(sorted_abs[:n//2 + 1]))
        candidates.append(set(sorted_abs[-(n//2 + 1):]))
        
        # Random samples
        rng = random.Random(42)
        for _ in range(30):
            sample = rng.sample(vals, n//2 + 1)
            candidates.append(set(sample))
        
        # Greedy with different orderings
        orderings = [
            sorted(vals, key=lambda x: -abs(x)),
            sorted(vals, key=lambda x: x),
            sorted(vals, key=lambda x: -x),
            sorted(vals, key=lambda x: abs(x)),
            sorted(vals, key=lambda x: random.Random(hash(x)).random()),
            sorted(vals, key=lambda x: random.Random(hash(x)+1).random()),
        ]
        
        for ordering in orderings:
            S = greedy_order(ordering)
            candidates.append(S)
        
        # Sort by size, largest first
        candidates.sort(key=len, reverse=True)
        
        best_S = set()
        best_size = -1
        
        for S in candidates:
            if time.time() - start_time > time_limit:
                break
            
            if not is_sum_free(S):
                continue
            
            S_improved = local_search(S)
            
            if not is_sum_free(S_improved):
                continue
                
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
        
        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 70.0
            
        S = solve_instance(vals, t_lim)
        
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
