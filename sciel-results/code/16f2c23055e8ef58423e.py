import time

def construct():
    N = 123457
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    # lpf[i] will hold the largest prime factor of i
    lpf = [0] * (N + 1)
    
    # Sieve-like loop to fill lpf
    # We iterate i from 2 to N. If lpf[i] is 0, i is prime.
    # We then update all multiples of i. Since we go in increasing order,
    # the last update to lpf[j] will be from the largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # P(1) is defined as 1
    lpf[1] = 1
    
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    mid = N // 2
    
    # Exhaustive search for the pair (a, b) with a+b=N minimizing max(P(a), P(b))
    for a in range(1, mid + 1):
        b = N - a
        # Calculate the score for this pair
        score = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
