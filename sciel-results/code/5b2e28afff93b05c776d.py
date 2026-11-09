import time
import random

def is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin
    
    # Target: minimize distinct primes.
    # Try sets of primes P.
    # Start with small P.
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_result = None
    best_score = float('inf')
    
    for P in prime_sets_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Greedy construction
        # We want 12 distinct integers <= 10000.
        # To maximize chance of finding a solution, we can try to find numbers
        # that are "close" to each other or follow a pattern.
        
        # Strategy: Randomized Greedy with multiple restarts
        found = False
        for restart in range(20): # Max 20 restarts per prime set
            if time.time() - start_time > timeout:
                break
                
            current_set = []
            # Pick a random starting number
            # To keep sums small and smooth, maybe pick small numbers?
            # Or numbers that are multiples of product(P)?
            
            # Let's try picking numbers from a specific range
            # If P={2,3,5,7}, product=210.
            # If a_i are multiples of 210, sums are multiples of 210.
            # Then we just need k_i + k_j to be smooth.
            
            # Let's try a simpler heuristic:
            # Generate candidate numbers 1..10000
            # Sort them by how many "smooth sums" they form with random samples?
            # Too slow.
            
            # Let's try constructing from a base set.
            # If we pick a_i = i * M, where M is product(P)?
            # Sums = (i+j)M. Primes of (i+j) + Primes of M.
            # If M contains P, we need i+j to be P-smooth.
            
            M = 1
            for p in P:
                M *= p
                
            # We need 12 distinct i such that i+j is P-smooth for all i!=j.
            # This is a known problem: finding a set of integers with smooth pairwise sums.
            # For small P, such sets exist.
            
            # Let's try to find such a set of indices k_1..k_12.
            # Then a_i = k_i * M.
            # Check if a_i <= 10000.
            
            # Greedy for k_i:
            k_set = []
            # Start with k=1?
            # If we pick k_i randomly, we can check smoothness of k_i + k_j.
            
            # Let's try a specific known structure or random search for k_i.
            # Random search for k_i in 1..(10000//M)
            max_k = 10000 // M
            if max_k < 12:
                continue
                
            # Greedy:
            # Pick k_1 randomly.
            # Pick k_2 such that k_1+k_2 is smooth.
            # ...
            
            # To speed up, precompute smooth numbers up to 2*max_k?
            # 2*max_k is small if M is large.
            
            smooth_sums = set()
            limit = 2 * max_k
            for s in range(2, limit + 1):
                if is_smooth(s, P):
                    smooth_sums.add(s)
                    
            # Now find 12 distinct k_i in [1, max_k] such that k_i+k_j in smooth_sums.
            # This is a clique problem in a graph where nodes are k_i and edges exist if sum is smooth.
            # We can use a simple backtracking or greedy.
            
            # Greedy:
            # Try to build the set.
            # Sort candidates?
            
            # Let's try random restarts for the k-set construction.
            success = False
            for inner_restart in range(50):
                if time.time() - start_time > timeout:
                    break
                    
                k_set = []
                # Pick first k randomly
                k1 = random.randint(1, max_k)
                k_set.append(k1)
                
                # Pick subsequent k's
                for _ in range(11):
                    candidates = []
                    for k in range(1, max_k + 1):
                        if k in k_set:
                            continue
                        # Check against all existing
                        ok = True
                        for ek in k_set:
                            if (k + ek) not in smooth_sums:
                                ok = False
                                break
                        if ok:
                            candidates.append(k)
                            
                    if not candidates:
                        break
                    
                    # Pick one randomly or the one that maximizes future options?
                    # Random is fast.
                    k_next = random.choice(candidates)
                    k_set.append(k_next)
                    
                if len(k_set) == 12:
                    success = True
                    break
                    
            if success:
                # Construct a_i
                a_set = [k * M for k in k_set]
                # Verify distinctness and bounds
                if len(set(a_set)) == 12 and all(x <= 10000 for x in a_set):
                    # Verify prime factors
                    primes_found = set()
                    for i in range(12):
                        for j in range(i+1, 12):
                            s = a_set[i] + a_set[j]
                            # Factor s
                            temp = s
                            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                                while temp % p == 0:
                                    primes_found.add(p)
                                    temp //= p
                            if temp > 1:
                                # Prime factor not in small list, add it
                                primes_found.add(temp)
                                
                    score = len(primes_found)
                    if score < best_score:
                        best_score = score
                        best_result = a_set
                    found = True
                    break # Found a valid set for this P, move to next P or break?
                    # The task is to MINIMIZE the number of primes.
                    # If we found a set with score 4 (for P={2,3,5,7}), that's good.
                    # We should try smaller P first?
                    # P={2,3} -> M=6. max_k=1666.
                    # If we find a set with P={2,3}, score is likely 2 or 3.
                    # So we should iterate P in increasing order of size?
                    # My list is [2,3], [2,3,5], etc.
                    # So if we find one for [2,3], we break.
        
        if found:
            break
            
    if best_result is None:
        # Fallback: return a simple set, e.g., 1..12
        # This will have many primes, but it's valid.
        return list(range(1, 13))
        
    return best_result
