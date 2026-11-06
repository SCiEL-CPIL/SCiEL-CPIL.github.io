import math
import random
import time

def construct():
    deadline = time.time() + 80
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = {m: prime_factors(m) for m in moduli}
    primes = sorted(set(p for m in moduli for p in pf[m]))
    
    # Capacity for each prime is the prime itself
    capacity = {p: p for p in primes}
    
    best_set = []
    
    # Greedy with randomization
    for _ in range(2000):
        if time.time() > deadline - 5:
            break
            
        # Randomize order with bias towards fewer prime factors
        # Key: (num_prime_factors, random_noise)
        order = sorted(moduli, key=lambda m: (len(pf[m]), random.random()))
        
        current_set = []
        prime_count = {p: 0 for p in primes}
        
        for m in order:
            can_add = True
            for p in pf[m]:
                if prime_count[p] >= capacity[p]:
                    can_add = False
                    break
            if can_add:
                current_set.append(m)
                for p in pf[m]:
                    prime_count[p] += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == len(moduli): # Max possible
                break

    # Local Search: Try to add any remaining modulus
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        prime_count = {p: 0 for p in primes}
        for m in best_set:
            for p in pf[m]:
                prime_count[p] += 1
        
        for m in moduli:
            if m in best_set:
                continue
            can_add = True
            for p in pf[m]:
                if prime_count[p] >= capacity[p]:
                    can_add = False
                    break
            if can_add:
                best_set.append(m)
                for p in pf[m]:
                    prime_count[p] += 1
                improved = True

    # Assign Residues
    # For each prime p, assign distinct values to m in S_p
    # v[m][p] will store the residue modulo p
    v = {m: {} for m in best_set}
    
    for p in primes:
        S_p = [m for m in best_set if p in pf[m]]
        S_p.sort()
        for i, m in enumerate(S_p):
            v[m][p] = i
            
    # Construct r_m using CRT
    result = []
    for m in best_set:
        # System: r ≡ v[m][p] (mod p) for all p in pf[m]
        # Since p are distinct primes, we can use CRT
        # r = sum( v[m][p] * (M/p) * inv(M/p, p) ) mod M
        # where M = product of p in pf[m]
        M = 1
        for p in pf[m]:
            M *= p
            
        r = 0
        for p in pf[m]:
            Mi = M // p
            # Modular inverse of Mi mod p
            # Since p is prime, inv = Mi^(p-2) mod p
            # Or just brute force for small p
            inv = 1
            while (Mi * inv) % p != 1:
                inv += 1
            r = (r + v[m][p] * Mi * inv) % M
            
        # r is in [0, M). Since M divides m, r < m.
        # Canonical residue is r % m, which is just r.
        result.append([r, m])
        
    # Ensure non-empty and valid
    if not result:
        return [[0, 2]]
        
    return result
