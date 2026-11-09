"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Precompute factorizations
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
        
    factorizations = [None] * (n + 1)
    for i in range(2, n + 1):
        factorizations[i] = prime_factorize(i)
        
    def divides_product(a, b, c):
        for p, exp_a in factorizations[a].items():
            exp_bc = factorizations[b].get(p, 0) + factorizations[c].get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
        
    # Precompute bad_pairs[a] = set of (b,c) with b<c such that a|bc
    # Only if time permits
    bad_pairs = [set() for _ in range(n + 1)]
    if time.time() < start_time + 10:
        for a in range(2, n + 1):
            for b in range(2, n + 1):
                for c in range(b + 1, n + 1):
                    if (b * c) % a == 0:
                        bad_pairs[a].add((b, c))

    def can_add(S, new_elem):
        S_list = list(S)
        m = len(S_list)
        
        # 1. new_elem | a*b
        if bad_pairs[new_elem]:
            # Check if any pair in S is in bad_pairs[new_elem]
            # Optimization: If S is small, iterate pairs in S.
            # If bad_pairs is small, iterate bad_pairs.
            # Here we just iterate pairs in S for simplicity and correctness
            for i in range(m):
                a = S_list[i]
                for j in range(i + 1, m):
                    b = S_list[j]
                    min_ab = a if a < b else b
                    max_ab = b if a < b else a
                    if (min_ab, max_ab) in bad_pairs[new_elem]:
                        return False
        else:
            # Fallback if precomp failed
            for i in range(m):
                a = S_list[i]
                for j in range(i + 1, m):
                    b = S_list[j]
                    if divides_product(new_elem, a, b):
                        return False

        # 2. a | new_elem * b for any distinct a,b in S
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if i == j:
                    continue
                b = S_list[j]
                if divides_product(a, new_elem, b):
                    return False
        return True

    def is_valid_set(S):
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(m):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True

    # Get primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    best_set = set(primes)
    best_size = len(best_set)
    
    candidates = list(range(2, n + 1))
    
    # Local Search
    for restart in range(30):
        if time.time() > deadline - 5:
            break
            
        # Try to extend best_set
        current_set = set(best_set)
        improved = True
        while improved:
            improved = False
            for x in candidates:
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
                    break
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Random Greedy Restart
        current_set = set()
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        for x in shuffled:
            if can_add(current_set, x):
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)

    if not is_valid_set(best_set):
        return primes
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
