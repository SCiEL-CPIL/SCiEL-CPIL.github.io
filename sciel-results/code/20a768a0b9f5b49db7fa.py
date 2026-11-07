import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
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
    num_primes = len(primes)
    
    # Precompute square-free bitsets for all n in (210, max_value]
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        if temp > 1:
            bits |= (1 << prime_to_idx[temp])
        sqf_bits[n] = bits
    
    # Target: square-free part of 210 = 2*3*5*7
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])
    
    # Incremental GF(2) basis building
    # basis: dict mapping pivot_bit -> (row_bits, combo_mask)
    # combo_mask is a big int where bit i is 1 if number (211+i) is in the combination
    basis = {}
    
    # Process numbers in increasing order
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            break
        
        v = sqf_bits[M]
        idx = M - 211  # 0-based index
        comb = 1 << idx
        
        # Reduce v by existing basis
        curr_v = v
        curr_comb = comb
        
        # Find highest set bit and reduce
        while curr_v != 0:
            hb = curr_v.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                curr_v ^= b_vec
                curr_comb ^= b_combo
            else:
                basis[hb] = (curr_v, curr_comb)
                break
        
        # After reduction, check if target is now in span
        # Reduce target using current basis
        t_curr = target_bits
        t_comb = 0  # We want to express target as combo of basis vectors
        
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                break
        
        if t_curr == 0:
            # Target is in span! Extract solution
            sol_mask = t_comb
            # Count bits in sol_mask
            sol_count = bin(sol_mask).count('1')
            
            if 1 <= sol_count <= max_size:
                # Extract the actual numbers
                result = []
                for i in range(M - 210):  # numbers from 211 to M
                    if (sol_mask >> i) & 1:
                        result.append(211 + i)
                
                # Verify
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(result)
                else:
                    # Verification failed, continue searching
                    continue
            
            # If solution too large, continue (might find smaller later)
            # But for minimal M, this shouldn't happen often
    
    # Fallback to known solution
    return [840]
