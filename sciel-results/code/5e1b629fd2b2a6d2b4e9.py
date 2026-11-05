import time

def construct():
    N = 123457
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve to find largest prime factor
    # Iterate i from 2 to N
    # If lpf[i] == 0, i is prime
    # For each prime i, update lpf for all multiples of i
    # Since we iterate i in increasing order, the last prime to divide a number 
    # is its largest prime factor.
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Find the pair (a, b) with a+b=N that minimizes max(lpf[a], lpf[b])
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We can stop early if we find a score of 2? 
    # P(n) is at least 2 for n>1. 
    # If we find max(P(a), P(b)) == 2, it's optimal (since P(1)=1, but a,b > 0).
    # Wait, if a=1, P(a)=1. If b is a power of 2, P(b)=2. Max is 2.
    # Can we get 1? Only if a=1 and b=1, sum=2. Here sum=123457.
    # So min possible score is 2 (if one is 1 and other is 2^k) or higher.
    
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
            
        b = N - a
        score = max(lpf[a], lpf[b])
        
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
            # If we found the theoretical minimum, we can break early
            # The minimum possible value for P(n) for n>1 is 2.
            # So max(P(a), P(b)) >= 2.
            if best_score == 2:
                break
                
    return [best_a, best_b]
