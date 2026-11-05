def construct():
    import random
    import time

    def compute_sizes(A):
        sums = set()
        prods = set()
        for i, a in enumerate(A):
            for j, b in enumerate(A):
                sums.add(a + b)
                prods.add(a * b)
        return len(sums), len(prods)
    
    def score(A):
        s, p = compute_sizes(A)
        return max(s, p)

    def greedy_build(start_val, rng, max_val=10000, target_size=20):
        A = [start_val]
        A_set = {start_val}
        # Pre-compute current sums and prods? 
        # It's easier to just recompute or maintain incrementally.
        # For N=20, recomputing is O(400) per candidate. 10000 candidates * 400 = 4M ops.
        # 20 steps * 4M = 80M ops per build.
        # 1000 builds = 80B ops. Too slow in Python.
        
        # Optimization: 
        # We don't need to check all 10000 candidates.
        # We can sample candidates or restrict to a smaller pool?
        # Or, we can use a smarter heuristic.
        
        # Let's try a hybrid: 
        # 1. Generate a pool of "interesting" numbers (e.g., small numbers, powers, primes).
        # 2. Greedily pick from this pool.
        
        pass

    def local_search(A, max_time=5.0):
        best_A = A[:]
        best_score = score(best_A)
        start_time = time.time()
        rng = random.Random(42)
        
        while time.time() - start_time < max_time:
            improved = False
            # Try single swaps
            for i in range(20):
                old_val = best_A[i]
                for _ in range(50):
                    new_val = rng.randint(1, 10000)
                    if new_val == old_val or new_val in best_A:
                        continue
                    best_A[i] = new_val
                    s_val = score(best_A)
                    if s_val < best_score:
                        best_score = s_val
                        improved = True
                    else:
                        best_A[i] = old_val
                if improved:
                    break
            if not improved:
                # Perturbation
                idxs = rng.sample(range(20), 3)
                old_vals = [best_A[i] for i in idxs]
                for i in idxs:
                    best_A[i] = rng.randint(1, 10000)
                if len(set(best_A)) < 20:
                    for k, i in enumerate(idxs):
                        best_A[i] = old_vals[k]
                    continue
                s_val = score(best_A)
                if s_val < best_score:
                    best_score = s_val
                else:
                    for k, i in enumerate(idxs):
                        best_A[i] = old_vals[k]
        return best_A, best_score

    # New Approach: Greedy Forward Construction with a restricted candidate pool
    # The candidate pool should be numbers that are "multiplicatively smooth" or small.
    # Numbers with many factors tend to have smaller product sets.
    
    def generate_candidate_pool():
        # Include small integers
        pool = set(range(1, 100))
        # Include powers of 2, 3, 5, 7
        for p in [2, 3, 5, 7, 11]:
            v = 1
            while v <= 10000:
                pool.add(v)
                v *= p
        # Include products of small primes
        for i in range(1, 50):
            for j in range(1, 50):
                val = i * j
                if val <= 10000:
                    pool.add(val)
        return list(pool)

    def greedy_build_from_pool(pool, rng, target_size=20):
        # Start with a random element from pool
        start_val = rng.choice(pool)
        A = [start_val]
        A_set = {start_val}
        
        # To speed up, we can pre-compute sums/prods of current A
        # But updating them incrementally is complex. 
        # Given N=20, let's just score the new set.
        # To limit candidates, we can sample 100 candidates from pool at each step.
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_score = float('inf')
            
            # Sample candidates
            # If pool is large, sample 200
            candidates = rng.sample(pool, min(200, len(pool)))
            
            for x in candidates:
                if x in A_set:
                    continue
                new_A = A + [x]
                # Check distinctness
                if len(set(new_A)) < len(new_A):
                    continue
                sc = score(new_A)
                if sc < best_score:
                    best_score = sc
                    best_candidate = x
            
            if best_candidate is None:
                # Fallback: pick any valid x
                for x in pool:
                    if x not in A_set:
                        best_candidate = x
                        break
                if best_candidate is None:
                    return None, float('inf')
            
            A.append(best_candidate)
            A_set.add(best_candidate)
        
        return A, score(A)

    pool = generate_candidate_pool()
    rng = random.Random(123)
    
    best_overall_A = None
    best_overall_score = float('inf')
    
    # Run multiple greedy builds
    for i in range(200):
        A, sc = greedy_build_from_pool(pool, rng)
        if A is None:
            continue
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_A = A[:]
        
        # Polish with local search
        if i % 5 == 0: # Polish every 5th build
            A_polished, sc_polished = local_search(A, max_time=0.5)
            if sc_polished < best_overall_score:
                best_overall_score = sc_polished
                best_overall_A = A_polished[:]

    if best_overall_A is None:
        return list(range(1, 21))
        
    return sorted(best_overall_A)
