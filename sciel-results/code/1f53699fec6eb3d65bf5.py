import time

def construct():
    S = 123457
    # Safety margin: evaluator kills at 90s. We use 80s to be safe.
    deadline = time.time() + 80.0
    
    # Compute largest prime factor for all numbers up to S
    # P[n] will hold the largest prime factor of n
    P = [1] * (S + 1)
    
    # Sieve for largest prime factors
    # If P[i] is still 1, then i is prime (no smaller prime factor found yet, 
    # and since we iterate in increasing order, no factor exists other than itself)
    for i in range(2, S + 1):
        if P[i] == 1:  # i is prime
            for j in range(i, S + 1, i):
                P[j] = i
                
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # Linear search over all pairs (a, S-a)
    # We only need to check a <= S/2 due to symmetry
    P_local = P
    S_local = S
    
    for a in range(1, S_local // 2 + 1):
        # Check deadline
        if time.time() > deadline:
            break
        
        b = S_local - a
        pa = P_local[a]
        pb = P_local[b]
        
        # The objective is to minimize max(P(a), P(b))
        score = pa if pa > pb else pb
        
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
            # Theoretical minimum for max(P(a), P(b)) is 2 (if both are powers of 2)
            # or 1 (if one is 1 and the other is a power of 2, but P(1)=1, P(2^k)=2, max=2)
            # If we reach 2, we can break early as it's the smallest prime.
            if best_score == 2:
                break

    return [best_a, best_b]
