"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # We want to find distinct integers a, b > 210, <= 10210
    # such that 210 * a * b is a perfect square.
    # This is equivalent to sf(210 * a) == sf(b).
    # We minimize max(a, b) - 210.
    
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We need primes up to sqrt(210 * 10210) ~ 1464 for sf(210*n)
    # and up to 10210 for sf(n)
    primes = sieve_primes(10210)
    
    def compute_squarefree(n, primes):
        result = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                result *= p
        if temp > 1:
            result *= temp
        return result
    
    # Precompute sf(n) for n in 211..10210
    sf_n = {}
    for n in range(211, 10211):
        sf_n[n] = compute_squarefree(n, primes)
        
    # Precompute sf(210*n) for n in 211..10210
    # 210*n is up to 2144100. sqrt is 1464.
    # We can reuse the primes list, it's fine.
    sf_210n = {}
    for n in range(211, 10211):
        val = 210 * n
        sf_210n[n] = compute_squarefree(val, primes)
        
    # Map sf(210*a) -> list of a
    # We want to find a, b such that sf(210*a) == sf(b) and a != b.
    # To minimize max(a,b), we can iterate M from 211 upwards.
    # For each M, check if there is a pair with max(a,b) = M.
    # This is O(N^2) in worst case if we do it naively.
    # Better: 
    # For each b, we have key_b = sf(b).
    # We need a such that sf(210*a) == key_b and a != b.
    # Let's build a map: key -> list of a's.
    a_by_key = {}
    for a in range(211, 10211):
        key = sf_210n[a]
        if key not in a_by_key:
            a_by_key[key] = []
        a_by_key[key].append(a)
        
    best_max = float('inf')
    best_pair = None
    
    # Iterate b from 211 to 10210
    # For each b, look up sf(b) in a_by_key
    # If found, iterate through a's.
    # Update best if max(a,b) is smaller.
    for b in range(211, 10211):
        key_b = sf_n[b]
        if key_b in a_by_key:
            for a in a_by_key[key_b]:
                if a != b:
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_pair = (a, b)
                        
    if best_pair:
        return [best_pair[0], best_pair[1]]
        
    # Fallback
    return [240, 224]

# EVOLVE-BLOCK-END
