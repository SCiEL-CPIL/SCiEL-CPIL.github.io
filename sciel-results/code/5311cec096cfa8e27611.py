import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute prime factor sets for sums
    # We only need to check membership in a specific prime set, so we can check on the fly or precompute a boolean array for specific P.
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_result = None
    best_score = float('inf')
    
    # Try prime sets of increasing size
    # Start with small sets to get a low score quickly
    for num_primes in range(2, 8):
        if time.time() > deadline - 2:
            break
            
        # Only try a subset of prime combinations to save time
        # Focus on small primes
        primes_to_consider = small_primes[:num_primes + 2]
        
        # Generate combinations of primes
        # To avoid explosion, we limit the number of combinations we test per size
        # For size 2, 3, 4 there are few. For 5+, we sample.
        if num_primes <= 4:
            prime_combos = list(combinations(primes_to_consider, num_primes))
        else:
            # Sample some combinations, prioritizing small primes
            prime_combos = list(combinations(primes_to_consider, num_primes))[:20]
            
        for prime_set in prime_combos:
            if time.time() > deadline - 2:
                break
                
            ps = set(prime_set)
            
            # Function to check if n is ps-smooth
            def is_smooth(n):
                while n > 1:
                    p = spf[n]
                    if p not in ps:
                        return False
                    while n % p == 0:
                        n //= p
                return True
            
            # Build a candidate pool
            # Strategy: Use a smaller pool to make clique search fast.
            # Start with 1..200, then 1..500, then 1..1000
            for limit in [200, 500, 1000, 2000, 5000, 10000]:
                if time.time() > deadline - 2:
                    break
                    
                candidates = list(range(1, limit + 1))
                
                # Precompute compatibility matrix? No, too big.
                # Precompute "smooth sums" for this specific P?
                # Let's just filter candidates:
                # A candidate x is good if it has many smooth sums with other candidates.
                
                # Greedy Clique Search
                # We want a clique of size 12.
                
                # Optimization: Precompute for each x, the list of y such that x+y is smooth.
                # This is expensive if done for all x, y.
                # Instead, do it on the fly but restrict candidates.
                
                # Let's try a randomized greedy approach with backtracking (limited)
                # Or just multiple greedy attempts.
                
                found = None
                attempts = 100 if limit <= 500 else 20
                
                for _ in range(attempts):
                    if time.time() > deadline - 2:
                        break
                        
                    # Random shuffle of candidates for diversity
                    import random
                    random.shuffle(candidates)
                    
                    current_set = []
                    # Start with a random candidate
                    start_idx = random.randint(0, len(candidates)-1)
                    current_set.append(candidates[start_idx])
                    used = {candidates[start_idx]}
                    
                    # Greedily add compatible numbers
                    for _ in range(11):
                        best_candidate = None
                        best_score_c = -1
                        
                        # Filter candidates compatible with ALL in current_set
                        compatible = []
                        for y in candidates:
                            if y in used:
                                continue
                            ok = True
                            for x in current_set:
                                if not is_smooth(x + y):
                                    ok = False
                                    break
                            if ok:
                                compatible.append(y)
                                
                        if not compatible:
                            break
                            
                        # Score: number of connections within compatible set?
                        # Just pick the one with the most connections to other compatible ones
                        # This is O(|compatible|^2)
                        # If |compatible| is small, this is fast.
                        
                        if len(compatible) < 12 - len(current_set):
                            break
                            
                        # Heuristic: Pick the one that allows the most future options
                        # Approximate by degree in the subgraph induced by 'compatible'
                        # To save time, just pick the first one? No, that's bad.
                        # Pick the one with highest "local density".
                        
                        # Optimization: If compatible is large, sample it
                        if len(compatible) > 50:
                            sample = random.sample(compatible, 50)
                        else:
                            sample = compatible
                        
                        for y in sample:
                            count = 0
                            for z in compatible:
                                if z != y and is_smooth(y + z):
                                    count += 1
                            if count > best_score_c:
                                best_score_c = count
                                best_candidate = y
                                
                        if best_candidate is None:
                            break
                            
                        current_set.append(best_candidate)
                        used.add(best_candidate)
                        
                    if len(current_set) == 12:
                        # Verify
                        valid = True
                        for i in range(12):
                            for j in range(i+1, 12):
                                if not is_smooth(current_set[i] + current_set[j]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            found = sorted(current_set)
                            break
                            
                if found:
                    # Calculate actual distinct primes
                    all_primes = set()
                    for i in range(12):
                        for j in range(i+1, 12):
                            n = found[i] + found[j]
                            while n > 1:
                                p = spf[n]
                                all_primes.add(p)
                                while n % p == 0:
                                    n //= p
                    score = len(all_primes)
                    if score < best_score:
                        best_score = score
                        best_result = found
                    # If we found a solution with score equal to num_primes, it's optimal for this P
                    if score <= num_primes:
                        return best_result
                    # If we found a good score, we might still find better with different P
                    # But if limit was small, we might have missed a better config in larger range?
                    # Usually smaller numbers are denser.
                    # Break out of limit loop if we found a solution?
                    # Yes, larger limits are slower. If we have a solution, try to improve it with different P sets, but maybe skip larger limits for this P.
                    break 

    if best_result is not None:
        return best_result
        
    # Fallback
    return list(range(1, 13))
