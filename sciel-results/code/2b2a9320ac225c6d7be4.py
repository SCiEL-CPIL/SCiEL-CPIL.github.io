import math
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0 # Safety margin
    
    N_MAX = 10210
    N_MIN = 211
    
    # 1. Sieve of Eratosthenes to find primes and smallest prime factors
    spf = list(range(N_MAX + 1))
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    primes = [i for i in range(2, N_MAX + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    # 2. Precompute bitmasks for all numbers in range
    # num_bitsets[x] is the vector of parity of exponents
    num_bitsets = {}
    for x in range(N_MIN, N_MAX + 1):
        temp = x
        b = 0
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                b |= (1 << prime_to_idx[p])
        num_bitsets[x] = b
        
    # 3. Compute target bitmask for 210
    target = 0
    temp = 210
    while temp > 1:
        p = spf[temp]
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        if count % 2 == 1:
            target |= (1 << prime_to_idx[p])
            
    # 4. Gaussian Elimination over GF(2)
    # basis[pivot] = (vector, mask)
    # mask is a bitmask where bit i is set if number (N_MIN + i) is in the combination
    basis = {}
    
    target_reduced = target
    target_mask = 0
    
    # We iterate M from N_MIN to N_MAX
    # We need to check if target is in span of {v(N_MIN), ..., v(M)}
    
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > DEADLINE:
            break
            
        v = num_bitsets[M]
        m = 1 << (M - N_MIN)
        
        # Reduce v using current basis
        while v != 0:
            pivot_pos = v.bit_length() - 1
            if pivot_pos in basis:
                bv, bm = basis[pivot_pos]
                v ^= bv
                m ^= bm
            else:
                break
        
        if v != 0:
            # v is a new basis vector
            pivot_pos = v.bit_length() - 1
            basis[pivot_pos] = (v, m)
            
            # Update target_reduced if it has this pivot
            if (target_reduced >> pivot_pos) & 1:
                target_reduced ^= v
                target_mask ^= m
        
        # Check if target is now in the span
        if target_reduced == 0:
            # Extract selected numbers
            selected = []
            # target_mask has bits set for numbers used
            # Note: The mask might contain numbers > M? No, we only added masks for numbers <= M.
            for i in range(M - N_MIN + 1):
                if (target_mask >> i) & 1:
                    selected.append(N_MIN + i)
            
            # Verify
            if selected:
                # Optional: Verify with square root
                # prod = 210
                # for x in selected:
                #     prod *= x
                # sq = math.isqrt(prod)
                # if sq * sq == prod:
                return [int(x) for x in selected]
            else:
                # If target was 0 (which it isn't, 210 is not a square), 
                # empty set would be solution. But 210 is not a square.
                # So selected should not be empty.
                pass
        
    # Fallback
    # If we didn't find a solution (should not happen), return a known valid set?
    # 840 is 210 * 4. 210*840 = 210^2 * 4 = (210*2)^2. 
    # But 840 > 210.
    # Is 840 in range? Yes.
    # Is it distinct? Yes, single element.
    # So [840] is always a valid solution.
    return [840]
