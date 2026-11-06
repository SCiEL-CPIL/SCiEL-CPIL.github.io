<<<<<<< SEARCH
    # Strategy 1: Clique Search for Small Prime Sets
    def try_find_clique_for_primes(P, max_attempts=500):
        # Generate P-smooth numbers up to 20000
        smooth_numbers = set()
        ps = list(P)
        def gen_smooth(val, idx):
            if val > 20000:
                return
            smooth_numbers.add(val)
            for i in range(idx, len(ps)):
                gen_smooth(val * ps[i], i)
        gen_smooth(1, 0)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            clique = [random.randint(1, 10000)]
            
            for _ in range(11):
                # Find candidates for next element
                candidate_sets = []
                for x_k in clique:
                    S_k = set()
                    for s in smooth_numbers:
                        v = s - x_k
                        if 1 <= v <= 10000:
                            S_k.add(v)
                    candidate_sets.append(S_k)
                
                candidates = candidate_sets[0]
                for i in range(1, len(candidate_sets)):
                    candidates = candidates.intersection(candidate_sets[i])
                
                candidates -= set(clique)
                
                if not candidates:
                    break
                
                next_k = random.choice(list(candidates))
                clique.append(next_k)
                
            if len(clique) == 12:
                if len(set(clique)) == 12 and all(1 <= x <= 10000 for x in clique):
                    primes_used = get_prime_factors_set(clique)
                    if primes_used.issubset(P):
                        return clique
        return None

    target_prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7}
    ]
    
    best_set = None
    best_score = float('inf')
    
    for P in target_prime_sets:
        if time.time() > deadline - 20:
            break
        clique = try_find_clique_for_primes(P, max_attempts=1000)
        if clique:
            score = evaluate_set(clique)
            if score < best_score:
                best_score = score
                best_set = clique
            if score <= 3:
                return best_set
=======
    # Strategy 1: Efficient Clique Search for Small Prime Sets
    def is_p_smooth(n, P):
        """Check if n is P-smooth (all prime factors in P)."""
        if n <= 1:
            return True
        for p in P:
            while n % p == 0:
                n //= p
        return n == 1
    
    def try_find_clique_for_primes(P, max_attempts=2000):
        """Greedy maximal clique heuristic with random restarts."""
        # Precompute: for each x in 1..10000, which y in 1..10000 have x+y P-smooth
        # We'll build this on-the-fly using a compatibility check
        
        def compatible(x, y, P):
            s = x + y
            if s > MAX_SUM:
                return False
            return is_p_smooth(s, P)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random start
            clique = [random.randint(1, 10000)]
            used = set(clique)
            
            for _ in range(11):
                # Find all candidates compatible with every element in clique
                candidates = []
                for v in range(1, 10001):
                    if v in used:
                        continue
                    if all(compatible(v, c, P) for c in clique):
                        candidates.append(v)
                
                if not candidates:
                    break
                
                # Prefer candidates that are compatible with many other unused numbers
                # (greedy heuristic to leave room for future picks)
                if len(candidates) > 20:
                    # Sample a subset to keep it fast
                    sample = random.sample(candidates, min(20, len(candidates)))
                else:
                    sample = candidates
                
                best_candidate = None
                best_score_local = -1
                for v in sample:
                    # Count how many other unused numbers are compatible with v
                    compat_count = 0
                    for w in range(1, 10001):
                        if w in used or w == v:
                            continue
                        if all(compatible(w, c, P) for c in clique) and compatible(v, w, P):
                            compat_count += 1
                            if compat_count > best_score_local:
                                break
                    if compat_count > best_score_local:
                        best_score_local = compat_count
                        best_candidate = v
                    if best_score_local >= 50:
                        break
                
                if best_candidate is None:
                    best_candidate = random.choice(candidates)
                
                clique.append(best_candidate)
                used.add(best_candidate)
            
            if len(clique) == 12 and len(set(clique)) == 12:
                if all(1 <= x <= 10000 for x in clique):
                    primes_used = get_prime_factors_set(clique)
                    if primes_used.issubset(P):
                        return clique
        return None

    # Also try a structured approach: build candidates from smooth numbers
    def try_structured_clique(P, max_attempts=500):
        """Generate P-smooth numbers and find clique among them."""
        smooth_numbers = set()
        ps = sorted(P)
        def gen_smooth(val, idx):
            if val > 20000:
                return
            smooth_numbers.add(val)
            for i in range(idx, len(ps)):
                gen_smooth(val * ps[i], i)
        gen_smooth(1, 0)
        
        # Filter to valid range
        smooth_in_range = [x for x in smooth_numbers if 1 <= x <= 10000]
        if len(smooth_in_range) < 12:
            return None
        
        random.shuffle(smooth_in_range)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            clique = []
            for x in smooth_in_range:
                if len(clique) >= 12:
                    break
                if all(is_p_smooth(x + c, P) for c in clique):
                    clique.append(x)
            
            if len(clique) == 12:
                primes_used = get_prime_factors_set(clique)
                if primes_used.issubset(P):
                    return clique
        return None

    target_prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7}
    ]
    
    best_set = None
    best_score = float('inf')
    
    for P in target_prime_sets:
        if time.time() > deadline - 20:
            break
        
        # Try structured approach first (faster)
        clique = try_structured_clique(P, max_attempts=200)
        if clique:
            score = evaluate_set(clique)
            if score < best_score:
                best_score = score
                best_set = clique
            if score <= 3:
                return best_set
        
        # Try greedy clique search
        clique = try_find_clique_for_primes(P, max_attempts=500)
        if clique:
            score = evaluate_set(clique)
            if score < best_score:
                best_score = score
                best_set = clique
            if score <= 3:
                return best_set
>>>>>>> REPLACE

Now let me also improve the local search to be more efficient with incremental evaluation:

