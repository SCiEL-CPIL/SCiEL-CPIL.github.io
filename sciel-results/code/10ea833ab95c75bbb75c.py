"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math

    # 1. Generate all primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # 2. Dynamic Programming
    # L[i]: max length of valid sequence ending at primes[i]
    # G[i]: min last gap among sequences of length L[i] ending at primes[i]
    # P[i]: predecessor index in primes
    L = [1] * n
    G = [0] * n
    P = [-1] * n
    
    for i in range(n):
        pi = primes[i]
        # Initialize for i (length 1, gap 0 is conceptually fine as no previous gap)
        # Actually, for i=0, L=1, G=0, P=-1 is correct.
        
        for j in range(i):
            pj = primes[j]
            d = pi - pj
            
            # Check if we can extend sequence ending at j
            # The last gap of the best sequence ending at j is G[j]
            # We need G[j] <= d
            if G[j] <= d:
                new_len = L[j] + 1
                if new_len > L[i]:
                    L[i] = new_len
                    G[i] = d
                    P[i] = j
                elif new_len == L[i]:
                    # If lengths are equal, prefer the smaller gap d
                    # to allow for easier future extensions
                    if d < G[i]:
                        G[i] = d
                        P[i] = j
                    # If d == G[i], we can keep the existing predecessor
                    # or update it; it doesn't strictly matter for length,
                    # but keeping the one with smaller gap is already handled.
                    # If d > G[i], we do not update because G[i] is smaller,
                    # which is better for future steps.

    # 3. Find the end of the longest sequence
    max_len = max(L)
    end_idx = L.index(max_len)
    
    # 4. Reconstruct the sequence
    seq = []
    cur = end_idx
    while cur != -1:
        seq.append(primes[cur])
        cur = P[cur]
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
