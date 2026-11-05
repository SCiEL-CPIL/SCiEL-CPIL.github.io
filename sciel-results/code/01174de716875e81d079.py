import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Primes up to ~600. 
    # Product of first 15 primes is ~6.15 * 10^17.
    # Product of first 16 primes is ~3.29 * 10^19 (too big for square-free).
    # So max distinct primes is 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors_fast(prime_powers):
        """Get sorted divisors from prime factorization as list of (prime, exponent) tuples."""
        divs = [1]
        for p, e in prime_powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        divs.sort()
        return divs
    
    def calc_score_from_divisors(divs):
        if len(divs) < 2:
            return 0
        score = 0
        # Optimization: only check gcd for pairs that could be coprime?
        # Actually gcd is fast in Python (C implementation).
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    def evaluate_factorization(powers):
        """powers: list of (prime, exponent). Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            if e > 0:
                # Early termination if n exceeds limit
                if n > 10**18 // (p ** e):
                    return -1, -1
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        
        # Check if time is running out
        if time.time() > deadline:
            return -1, -1

        divs = get_divisors_fast(powers)
        s = calc_score_from_divisors(divs)
        return s, n

    best_n = 2
    best_score = 0
    
    # Generate initial candidates
    # We want to explore the space of exponent vectors e = [e1, e2, ..., ek]
    # such that e1 >= e2 >= ... >= ek >= 1 and product(p_i^e_i) <= 10^18.
    
    # Strategy:
    # 1. Start with square-free numbers (all e_i = 1) for k up to 15.
    # 2. Try to increase exponents of smaller primes while decreasing or keeping larger ones.
    # 3. Use local search to refine.
    
    candidates = []
    
    # Add square-free candidates
    for k in range(2, 16):
        pows = [(primes[i], 1) for i in range(k)]
        candidates.append(pows)
        
    # Add some structured candidates with higher exponents on small primes
    # e.g., 2^a * 3^b * 5^c ...
    # We'll generate these via a simple DFS that respects monotonicity and limit.
    
    def dfs_exponents(idx, current_prod, current_exps):
        if time.time() > deadline:
            return
        if idx >= len(primes):
            # Reached end of primes, if we have at least 2 factors, add candidate
            if len(current_exps) >= 2 and current_prod >= 2:
                candidates.append([(primes[i], current_exps[i]) for i in range(len(current_exps))])
            return
            
        p = primes[idx]
        # Max exponent for this prime given current_prod
        # p^e <= 10^18 / current_prod
        max_e = 0
        temp = current_prod
        while temp * p <= 10**18:
            temp *= p
            max_e += 1
            
        # If we are at the first prime, we can try higher exponents
        # To keep the search space manageable, we limit the depth and branching
        # We only do this for the first few primes to avoid explosion
        
        if idx < 5:
            for e in range(0, min(max_e, 15) + 1):
                # Monotonicity constraint: e <= previous exponent (if any)
                # Actually, for minimal n for a given tau, we want e_i >= e_{i+1}.
                # Here we are building n, so we choose e_i.
                # To ensure we cover "highly composite" like structures, we enforce e_i >= e_{i+1}.
                # But we don't know e_{i+1} yet.
                # Let's just do a broad search for small idx.
                
                new_prod = current_prod * (p ** e)
                if new_prod > 10**18:
                    break
                current_exps.append(e)
                dfs_exponents(idx + 1, new_prod, current_exps)
                current_exps.pop()
        else:
            # For larger primes, only try e=0 or e=1 to keep it square-free-ish or simple
            # This prevents combinatorial explosion
            for e in [0, 1]:
                if e == 0 and idx > 0 and current_exps and current_exps[-1] == 0:
                    # If we stop adding primes, we can stop recursion
                    # But we need to check if current state is a candidate
                    if len(current_exps) >= 2 and current_prod >= 2:
                        candidates.append([(primes[i], current_exps[i]) for i in range(len(current_exps))])
                    return
                new_prod = current_prod * (p ** e)
                if new_prod > 10**18:
                    continue
                current_exps.append(e)
                dfs_exponents(idx + 1, new_prod, current_exps)
                current_exps.pop()
                
    # Run DFS to generate candidates
    # We limit the depth to first 15 primes
    dfs_exponents(0, 1, [])
    
    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        # Normalize: remove trailing zeros
        while c and c[-1][1] == 0:
            c.pop()
        if not c:
            continue
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Evaluate all candidates
    for pows in unique_candidates:
        if time.time() > deadline:
            break
        s, n = evaluate_factorization(pows)
        if s > best_score and n > 0:
            best_score = s
            best_n = n

    # Local Search / Hill Climbing
    # We refine the best candidates found.
    # We take the top few distinct structures and try to perturb them.
    
    # Store top candidates
    top_candidates = []
    for pows in unique_candidates:
        if time.time() > deadline - 5:
            break
        s, n = evaluate_factorization(pows)
        if s > 0:
            top_candidates.append((s, pows))
            
    # Sort by score descending
    top_candidates.sort(key=lambda x: -x[0])
    
    # Take top 20 candidates for local search
    for s, pows in top_candidates[:20]:
        if time.time() > deadline - 10:
            break
            
        current_pows = [list(p) for p in pows]
        current_score = s
        current_n = 1
        for p, e in current_pows:
            current_n *= p ** e
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try all possible moves
            # 1. Increase exponent of prime i
            # 2. Decrease exponent of prime i
            # 3. Shift exponent from i to i+1 (if i < k-1)
            
            k = len(current_pows)
            
            # Try adding a new prime factor (if last one is 1 or small)
            # This is handled by the DFS mostly, but let's try increasing existing.
            
            for i in range(k):
                p, e = current_pows[i]
                
                # Move: Increase e_i by 1
                if e < 20: # Arbitrary limit
                    new_pows = [list(x) for x in current_pows]
                    new_pows[i][1] += 1
                    # Check limit
                    new_n = current_n * p
                    if new_n <= 10**18:
                        ns, nn = evaluate_factorization(new_pows)
                        if ns > current_score:
                            current_score = ns
                            current_n = nn
                            current_pows = new_pows
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                            break # Restart search from new state
                    
                # Move: Decrease e_i by 1
                if e > 1:
                    new_pows = [list(x) for x in current_pows]
                    new_pows[i][1] -= 1
                    new_n = current_n // p
                    if new_n >= 2:
                        ns, nn = evaluate_factorization(new_pows)
                        if ns > current_score:
                            current_score = ns
                            current_n = nn
                            current_pows = new_pows
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                            break
                
                # Move: Shift 1 from e_i to e_{i+1} (if i < k-1 and e_{i+1} exists)
                if i < k - 1:
                    p_next, e_next = current_pows[i+1]
                    if e > 1:
                        new_pows = [list(x) for x in current_pows]
                        new_pows[i][1] -= 1
                        new_pows[i+1][1] += 1
                        new_n = current_n // p * p_next
                        if new_n <= 10**18 and new_n >= 2:
                            ns, nn = evaluate_factorization(new_pows)
                            if ns > current_score:
                                current_score = ns
                                current_n = nn
                                current_pows = new_pows
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_n = current_n
                                break
                            
                # Move: Shift 1 from e_{i+1} to e_i
                if i < k - 1:
                    p_next, e_next = current_pows[i+1]
                    if e_next > 1:
                        new_pows = [list(x) for x in current_pows]
                        new_pows[i][1] += 1
                        new_pows[i+1][1] -= 1
                        new_n = current_n // p_next * p
                        if new_n <= 10**18 and new_n >= 2:
                            ns, nn = evaluate_factorization(new_pows)
                            if ns > current_score:
                                current_score = ns
                                current_n = nn
                                current_pows = new_pows
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_n = current_n
                                break

    # Final check
    if best_n < 2:
        best_n = 2
        best_score = 1
        
    return best_n
