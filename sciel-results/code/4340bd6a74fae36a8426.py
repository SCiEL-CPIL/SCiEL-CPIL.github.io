import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 256
    candidates = list(range(2, n + 1))
    
    # Sieve for primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    prime_set = set(primes)
    
    # For each number, get its prime factors
    def get_prime_factors(x):
        factors = []
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
        return factors
    
    # Precompute prime factors for all numbers
    pf = {}
    for x in range(2, n + 1):
        pf[x] = get_prime_factors(x)
    
    # Key insight: if prime p is in S, then no other multiple of p can be in S.
    # So for each prime p, we either keep p (and remove all other multiples of p)
    # or remove p (and can keep multiples of p).
    
    # Let's think of it as: for each prime p, decide if p is "active" (in S) or not.
    # If p is active, all other multiples of p are excluded.
    # Then among the remaining numbers, we need to check the full constraint.
    
    # Strategy: 
    # 1. Start with all primes (54 primes)
    # 2. For each prime p, consider removing it and adding its multiples
    # 3. Use local search to optimize
    
    def is_valid_set(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S."""
        if x in S:
            return True
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
                if (a * b) % x == 0:
                    return False
                if (x * a) % b == 0:
                    return False
        return True
    
    # Phase 1: Greedy construction with prime decisions
    # For each prime, decide to keep or remove based on net gain
    
    # Start with all primes
    S = set(primes)
    best_S = set(S)
    best_size = len(S)
    
    # Try removing each prime and adding its multiples
    for p in primes:
        if time.time() > deadline - 10:
            break
        
        # If we remove p, which multiples of p can we add?
        multiples_of_p = [x for x in range(2, n + 1) if x % p == 0 and x != p]
        
        # Try adding multiples of p one by one
        new_S = set(S)
        new_S.remove(p)
        
        added = 0
        for m in sorted(multiples_of_p):
            if can_add(new_S, m):
                new_S.add(m)
                added += 1
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S.copy()
        elif len(new_S) > len(S):
            S = new_S
    
    # Phase 2: Local search - try to add any remaining number
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in candidates:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                improved = True
    
    # Phase 3: Try removing elements and adding others (swap moves)
    for _ in range(50):  # Limited iterations
        if time.time() > deadline - 5:
            break
        # Try removing a random element and adding a better one
        if not S:
            break
        to_remove = random.choice(list(S))
        new_S = set(S)
        new_S.remove(to_remove)
        
        # Try to add as many as possible
        added_any = False
        for x in candidates:
            if x in new_S:
                continue
            if can_add(new_S, x):
                new_S.add(x)
                added_any = True
        
        if len(new_S) > len(S):
            S = new_S
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
    
    # Final validation and return
    result = list(best_S)
    result = [int(x) for x in result]
    return result
