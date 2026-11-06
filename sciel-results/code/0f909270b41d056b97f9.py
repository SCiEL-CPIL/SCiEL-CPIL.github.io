"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt

    deadline = time.time() + 55.0  # safety margin before 60s hard kill

    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    P = len(primes)
    MaxGap = N

    # DP tables
    # DP[j][g] = max length of valid sequence ending at primes[j] with last gap <= g
    DP = [[0] * (MaxGap + 1) for _ in range(P)]
    # prev_prime[j][g] = index of previous prime in the sequence that achieved DP[j][g]
    # Note: This stores the prev prime for the *specific* gap that achieved the prefix max.
    prev_prime = [[-1] * (MaxGap + 1) for _ in range(P)]
    # argmax_prefix[j][g] = the specific gap h <= g that achieved the max length DP[j][g]
    argmax_prefix = [[-1] * (MaxGap + 1) for _ in range(P)]

    # Initialize for single elements
    # Any single prime is a valid sequence of length 1.
    # We handle this in the loop by initializing max_val = 1.

    for j in range(P):
        p_j = primes[j]
        # temp[g] will store the max length of a sequence ending at p_j with *exact* last gap g
        # derived from extending a sequence ending at some p_i.
        temp = [0] * (MaxGap + 1)
        prev_temp = [-1] * (MaxGap + 1)

        for i in range(j):
            p_i = primes[i]
            g = p_j - p_i
            # g is always <= 10000
            # We can extend any sequence ending at p_i with last gap <= g
            val = DP[i][g] + 1
            if val > temp[g]:
                temp[g] = val
                prev_temp[g] = i

        # Compute prefix max for p_j
        max_val = 1
        best_gap = -1
        prev_argmax = -1
        
        for g in range(MaxGap + 1):
            if temp[g] > max_val:
                max_val = temp[g]
                best_gap = g
                prev_argmax = prev_temp[g]
            
            DP[j][g] = max_val
            argmax_prefix[j][g] = best_gap
            prev_prime[j][g] = prev_argmax

    # Find the best sequence
    best_len = 0
    best_j = -1
    best_g = -1
    
    for j in range(P):
        # The max length for prime j is DP[j][MaxGap]
        if DP[j][MaxGap] > best_len:
            best_len = DP[j][MaxGap]
            best_j = j
            best_g = MaxGap

    # Backtrack to reconstruct the sequence
    path = []
    curr_j = best_j
    curr_g = best_g
    
    while curr_j != -1:
        path.append(primes[curr_j])
        h = argmax_prefix[curr_j][curr_g]
        if h == -1:
            break
        curr_j = prev_prime[curr_j][curr_g]
        curr_g = h
        
    path.reverse()
    
    return path
# EVOLVE-BLOCK-END
