import time

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Leave 20s margin for evaluator overhead
    
    # Upper bound for values we need LPF for is 1000000 + 1 = 1000001
    max_val = 1000001
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (max_val + 1)
    
    # Sieve to find largest prime factors
    # For each prime p, it is the largest prime factor for all multiples of p
    # that are not divisible by any prime > p.
    # Since we iterate p in increasing order, if we simply assign lpf[j] = p 
    # for all multiples j, the last assignment (largest p) will remain.
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
                
    # Time check
    if time.time() > deadline:
        return 10000

    best_n = 10000
    min_score = float('inf')
    
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    # We can early-terminate if we find a very low score
    # Scores 2 and 3 are impossible for n >= 10000
    # Score 5 would be excellent; score 7 is the next realistic target
    for n in range(10000, 1000001):
        # Check time periodically
        if n % 100000 == 0 and time.time() > deadline:
            break
            
        # The score is the largest prime factor of the product n(n+1)
        # which is max(lpf[n], lpf[n+1])
        score = max(lpf[n], lpf[n + 1])
        
        if score < min_score:
            min_score = score
            best_n = n
            
            # Early termination: if we find score 5, it's almost certainly optimal
            # (score 2,3 impossible; score 5 is very rare)
            if min_score <= 5:
                break
    
    return best_n
