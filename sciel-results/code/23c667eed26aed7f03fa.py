import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    N = 210
    MAX_X = 10210
    MAX_K = 100
    
    # 1. Sieve of Eratosthenes
    limit = MAX_X
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Precompute vectors for all x in (210, 10210]
    # v[x] is the integer bitset of prime exponents mod 2
    vecs = {}
    for x in range(211, MAX_X + 1):
        v = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                v |= (1 << prime_to_idx[temp])
        vecs[x] = v
        
    # Compute target vector for 210
    t = 210
    target_vec = 0
    for p in primes:
        if p * p > t:
            break
        if t % p == 0:
            count = 0
            while t % p == 0:
                t //= p
                count += 1
            if count % 2 == 1:
                target_vec |= (1 << prime_to_idx[p])
    if t > 1:
        if t in prime_to_idx:
            target_vec |= (1 << prime_to_idx[t])
            
    # 3. Incremental Span Check using Dictionary Basis
    basis = {}
    
    for Y in range(211, MAX_X + 1):
        if time.time() > deadline:
            break
            
        v_new = vecs[Y]
        hist_new = 1 << Y
        
        # Reduce v_new by existing basis (inlined for speed)
        v_curr = v_new
        h_curr = hist_new
        while v_curr:
            hsb = v_curr.bit_length() - 1
            if hsb in basis:
                b_vec, b_hist = basis[hsb]
                v_curr ^= b_vec
                h_curr ^= b_hist
            else:
                break
        
        if v_curr != 0:
            hsb = v_curr.bit_length() - 1
            basis[hsb] = (v_curr, h_curr)
            
        # Check if target_vec is in span (inlined for speed)
        v_t = target_vec
        h_t = 0
        while v_t:
            hsb = v_t.bit_length() - 1
            if hsb in basis:
                b_vec, b_hist = basis[hsb]
                v_t ^= b_vec
                h_t ^= b_hist
            else:
                break
        
        if v_t == 0:
            # Solution found in h_t
            selected = []
            ht = h_t
            while ht:
                lsb = (ht & -ht).bit_length() - 1
                selected.append(lsb)
                ht ^= (1 << lsb)
            
            if len(selected) <= MAX_K:
                # Verify
                prod = 210
                for x in selected:
                    prod *= x
                sqrt_prod = math.isqrt(prod)
                if sqrt_prod * sqrt_prod == prod:
                    return selected
                else:
                    pass

    # Fallback: return the seed solution
    return [840]
