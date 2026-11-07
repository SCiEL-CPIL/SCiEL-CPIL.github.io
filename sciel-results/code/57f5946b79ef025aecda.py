import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # Sieve primes up to max_value
    sieve = [True] * (max_value + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_value**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_value + 1, i):
                sieve[j] = False
    primes = [i for i in range(max_value + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute square-free bitsets using SPF for speed
    spf = list(range(max_value + 1))
    for i in range(2, int(max_value**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_value + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        sqf_bits[n] = bits
    
    # Target: square-free part of 210 = 2*3*5*7
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])
    
    # Binary search on M: find smallest M such that a valid subset exists in (210, M]
    def feasible(M):
        """Check if there's a non-empty subset of {211, ..., M} whose product with 210 is a square."""
        n_candidates = M - 210
        if n_candidates <= 0:
            return None
        
        # Build GF(2) basis incrementally and track target reduction
        basis = {}
        
        for idx in range(n_candidates):
            v = sqf_bits[211 + idx]
            comb = 1 << idx
            
            curr_v = v
            curr_comb = comb
            
            while curr_v != 0:
                hb = curr_v.bit_length() - 1
                if hb in basis:
                    b_vec, b_combo = basis[hb]
                    curr_v ^= b_vec
                    curr_comb ^= b_combo
                else:
                    basis[hb] = (curr_v, curr_comb)
                    break
        
        # Reduce target using current basis
        t_curr = target_bits
        t_comb = 0
        
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                break
        
        if t_curr == 0:
            sol_mask = t_comb
            sol_count = bin(sol_mask).count('1')
            if 1 <= sol_count <= max_size:
                return sol_mask
        return None
    
    # Binary search for minimal M
    lo = 211
    hi = max_value
    best_mask = None
    best_M = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        mask = feasible(mid)
        
        if mask is not None:
            best_mask = mask
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_mask is not None and best_M is not None:
        result = []
        for i in range(best_M - 210):
            if (best_mask >> i) & 1:
                result.append(211 + i)
        
        # Verify
        prod = base
        for x in result:
            prod *= x
        s = math.isqrt(prod)
        if s * s == prod:
            return list(result)
    
    # Fallback: linear scan from 211 upward
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            break
        
        v = sqf_bits[M]
        idx = M - 211
        comb = 1 << idx
        
        curr_v = v
        curr_comb = comb
        
        basis = {}
        # Rebuild basis for this M (inefficient but simple)
        for j in range(idx):
            vv = sqf_bits[211 + j]
            cc = 1 << j
            cv = vv
            ccb = cc
            while cv != 0:
                hb = cv.bit_length() - 1
                if hb in basis:
                    b_vec, b_combo = basis[hb]
                    cv ^= b_vec
                    ccb ^= b_combo
                else:
                    basis[hb] = (cv, ccb)
                    break
        
        cv = v
        ccb = comb
        while cv != 0:
            hb = cv.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                cv ^= b_vec
                ccb ^= b_combo
            else:
                basis[hb] = (cv, ccb)
                break
        
        t_curr = target_bits
        t_comb = 0
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                break
        
        if t_curr == 0:
            sol_mask = t_comb
            sol_count = bin(sol_mask).count('1')
            if 1 <= sol_count <= max_size:
                result = []
                for i in range(M - 210):
                    if (sol_mask >> i) & 1:
                        result.append(211 + i)
                
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(result)
    
    # Final fallback
    return [840]
