import math
import time

def construct():
    limit = 10210
    # Primes up to 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute parity bitmasks for all numbers in range
    bitmasks = {}
    for n in range(211, limit + 1):
        mask = 0
        x = n
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                count = 0
                while x % p == 0:
                    x //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << i)
        if x > 1:
            idx = prime_to_idx[x]
            mask |= (1 << idx)
        bitmasks[n] = mask
        
    # Target bitmask for 210
    target_mask = 0
    x = 210
    for i, p in enumerate(primes):
        if p * p > x:
            break
        if x % p == 0:
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                target_mask |= (1 << i)
    if x > 1:
        idx = prime_to_idx[x]
        target_mask |= (1 << idx)
        
    basis = {}
    
    start_time = time.time()
    
    for M in range(211, limit + 1):
        if time.time() - start_time > 55:
            return [840]
            
        v = bitmasks[M]
        u = v
        c = 1 << (M - 211)
        
        for pivot, b_vec, b_comb in basis.values():
            if (u >> pivot) & 1:
                u ^= b_vec
                c ^= b_comb
        
        if u != 0:
            pivot = u.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (pivot, u, c)
                
        t_u = target_mask
        t_c = 0
        for pivot, b_vec, b_comb in basis.values():
            if (t_u >> pivot) & 1:
                t_u ^= b_vec
                t_c ^= b_comb
        
        if t_u == 0:
            selected = []
            for i in range(M - 211 + 1):
                if (t_c >> i) & 1:
                    selected.append(211 + i)
            
            if len(selected) > 100:
                continue
                
            prod = 1
            for x in selected:
                prod *= x
            prod *= 210
            s = math.isqrt(prod)
            if s * s == prod:
                return [int(x) for x in selected]
                
    return [840]
