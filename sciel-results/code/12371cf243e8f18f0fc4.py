import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_score = 0
    best_set = []
    
    # Precompute smallest power of prime p >= L for various L?
    # Or just compute on the fly.
    
    # Primes up to 256
    primes = []
    for i in range(2, 257):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    for m in range(2, 257):
        if time.time() > deadline:
            break
            
        L = 256 // m + 1
        
        # Build V: smallest power of prime p >= L, p not dividing m, power <= 256
        V = []
        for p in primes:
            if m % p == 0:
                continue
            # Find smallest power p^k >= L
            val = 1
            while val < L:
                val *= p
                if val > 256:
                    break
            if val <= 256 and val >= L:
                V.append(val)
        
        if not V:
            # Only m
            score = 1.0 / m
            if score > best_score:
                best_score = score
                best_set = [m]
            continue
        
        # Split V into small (< 17) and large (>= 17)
        V_small = [v for v in V if v < 17]
        V_large = [v for v in V if v >= 17]
        
        # Sort V_small for consistency
        V_small.sort()
        
        # Iterate subsets of V_small
        n_small = len(V_small)
        # Max 2^7 = 128 subsets
        for mask in range(1 << n_small):
            A = []
            for i in range(n_small):
                if mask & (1 << i):
                    A.append(V_small[i])
            
            # Check if A is valid (pairwise product > 256)
            valid = True
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    if A[i] * A[j] <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check compatibility of A with m
            for u in A:
                if m * u <= 256:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Determine B from V_large
            B = []
            for v in V_large:
                # Check compatibility with m
                if m * v <= 256:
                    continue
                # Check compatibility with A
                compatible = True
                for u in A:
                    if u * v <= 256:
                        compatible = False
                        break
                if compatible:
                    B.append(v)
            
            # Calculate score
            score = 1.0 / m
            for u in A:
                score += 1.0 / u
            for v in B:
                score += 1.0 / v
            
            if score > best_score:
                best_score = score
                best_set = [m] + A + B
    
    return best_set
