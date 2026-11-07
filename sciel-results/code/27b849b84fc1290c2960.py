import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    MAX_VAL = 10210
    MIN_VAL = 211
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute bitmask V(n) for squarefree kernel
    # First, map primes to bit indices
    primes = [i for i in range(2, MAX_VAL + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    V = [0] * (MAX_VAL + 1)
    # V(1) = 0
    for n in range(2, MAX_VAL + 1):
        x = n
        mask = 0
        while x > 1:
            p = spf[x]
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            if exp % 2 == 1:
                mask ^= (1 << prime_to_idx[p])
        V[n] = mask
        
    target_mask = V[210]
    
    # 3. Search for minimal M with a pair (a, b)
    seen_masks = set()
    best_pair = None
    best_M = None
    
    for b in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        # Check if there exists a in (210, b) such that V(a) ^ V(b) == target_mask
        # => V(a) == V(b) ^ target_mask
        needed = V[b] ^ target_mask
        if needed in seen_masks:
            # We found a pair. We need to recover 'a'.
            # We didn't store 'a', only the mask.
            # We need to store the smallest 'a' for each mask to ensure distinctness and potentially smaller max?
            # Actually, we just need *any* valid 'a'.
            # To recover 'a', we should store mask -> a.
            # Let's refactor the loop to store mask -> a.
            pass 
            
        seen_masks.add(V[b])
        
    # The above loop doesn't recover 'a'. Let's fix it.
    mask_to_a = {}
    for b in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        needed = V[b] ^ target_mask
        if needed in mask_to_a:
            a = mask_to_a[needed]
            # Verify a != b (guaranteed by loop order a < b)
            # Verify constraints
            if a > 210 and b > 210 and a != b:
                # Check product
                prod = 210 * a * b
                sq = isqrt(prod)
                if sq * sq == prod:
                    return [a, b]
        
        # Store b in mask_to_a if not present (keep smallest a for each mask? 
        # It doesn't matter for existence, but keeping smallest a might help if we needed to optimize sum? 
        # We only care about max(b), so first found is best for this b.
        # For future b's, any a works.
        if V[b] not in mask_to_a:
            mask_to_a[V[b]] = b
            
    # Fallback if no pair found
    # Try to find a single number? Impossible since 210 is squarefree and >1.
    # Try to find a triple? 
    # Or just return a known valid solution.
    # 840 = 210 * 4. 210 * 840 = 210^2 * 4 = (210*2)^2. Square.
    # 840 is in range (210, 10210].
    return [840]
