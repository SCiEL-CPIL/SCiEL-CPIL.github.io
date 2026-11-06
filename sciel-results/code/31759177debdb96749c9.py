import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    N = 1000000
    N_MIN = 10000
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (N + 1)
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            lpf[i] = i
            for j in range(i, N + 1, i):
                lpf[j] = i
                if time.time() > deadline:
                    # Fallback if too slow (unlikely)
                    return 10000

    # Find best n
    best_n = N_MIN
    best_score = float('inf')
    
    # We can break early if we find a pair with score 2? 
    # Max(L(n), L(n+1)) = 2 implies both are powers of 2.
    # Consecutive powers of 2? 1, 2 (1 is not power of 2 in this context? 1 has no prime factors, LPF(1)=1? Or 0? 
    # Usually LPF(1) is defined as 1 or 0. 
    # Task: "largest prime factor". 1 has no prime factors. 
    # If n=1, n+1=2. LPF(1) is undefined or 1? 
    # The prompt says "local lower cutoff prevents the trivial pair 1,2". 
    # This implies n=1 is trivial. 
    # For n >= 10000, LPF is well-defined and >= 2.
    # The minimum possible LPF for any integer > 1 is 2.
    # Can we have max(LPF(n), LPF(n+1)) = 2?
    # This requires n and n+1 to be powers of 2.
    # The only consecutive powers of 2 are 1 and 2.
    # So for n >= 10000, the score must be at least 3? 
    # If one is a power of 2 (LPF=2) and the other is a power of 3 (LPF=3)?
    # 2^k and 3^m consecutive? 
    # 2 and 3 are consecutive. 
    # 8 and 9 are consecutive. LPF(8)=2, LPF(9)=3. Max=3.
    # 3 and 4 are consecutive. LPF(3)=3, LPF(4)=2. Max=3.
    # So score 3 is possible.
    # Are there pairs with score 2? No, because they must be powers of 2, and only 1,2 work.
    # So if we find a pair with score 3, it's the global minimum?
    # Wait, what if LPF(n)=2 and LPF(n+1)=2? Impossible for n>1.
    # What if LPF(n)=2 and LPF(n+1)=3? Max=3.
    # What if LPF(n)=3 and LPF(n+1)=2? Max=3.
    # So 3 is the theoretical minimum for n > 2.
    
    for n in range(N_MIN, N):
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            if best_score == 3:
                # Can't do better than 3 for n > 2
                break
                
    return best_n
