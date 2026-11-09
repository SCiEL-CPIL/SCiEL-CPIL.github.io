import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_val = 10210
    min_val = 211
    
    # We need masks for numbers 211..max_val
    # masks[i] will be the squarefree kernel of i represented as a bitset of primes
    # Bit p is set if prime p divides i an odd number of times
    
    # Initialize masks to 0
    masks = [0] * (max_val + 1)
    
    # Sieve for primes up to max_val
    # For each prime p, iterate multiples and toggle the bit if exponent is odd
    # To get exponent parity, we can just divide by p until not divisible, count steps.
    # Or, we can use a modified sieve.
    # For each i, if masks[i] is still 0 (unprocessed? no, 0 is a valid mask for squares), 
    # we need to know if i is a prime.
    
    # Let's stick to factoring each number, it's simple and fast enough for 10k.
    # But let's optimize the factoring loop.
    
    # Precompute smallest prime factor (SPF) for all numbers up to max_val
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute masks using SPF
    for i in range(min_val, max_val + 1):
        x = i
        mask = 0
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << p)
        masks[i] = mask
        
    # Target mask for 210 = 2 * 3 * 5 * 7
    target_mask = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    basis = {}
    
    # Iterate M from min_val to max_val
    for M in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
            
        v = masks[M]
        h = 1 << (M - min_val)
        
        # Insert v into basis
        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                basis[p] = (temp_v, temp_h)
                break
        
        # Check if target is in span
        t_v = target_mask
        t_h = 0
        while t_v:
            p = t_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                t_v ^= b_vec
                t_h ^= b_hist
            else:
                break
                
        if t_v == 0:
            # Solution found
            sol_indices = []
            hh = t_h
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(min_val + idx)
                hh ^= lsb
                
            sol = sol_indices
            if len(sol) <= 100:
                # Verify
                prod = 210
                for x in sol:
                    prod *= x
                if _is_square(prod):
                    return [int(x) for x in sol]
            # If len > 100, continue to next M? 
            # Theoretically, a larger M might allow a smaller set?
            # No, the set must be from {211..M}. If the span at M contains a small set, 
            # the basis reduction might have found a large one. 
            # However, for the *first* M where target is in span, the dimension is minimal.
            # So the solution size should be minimal in some sense.
            # If it's > 100, it's a problem. But unlikely.
            pass
            
    # Fallback
    return [224, 240]
