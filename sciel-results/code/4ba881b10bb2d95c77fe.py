import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1

    # Step 1: Sieve of Eratosthenes to get Smallest Prime Factor (SPF)
    # This allows O(log x) factorization for any x
    spf = list(range(MAX_VAL + 1))
    if MAX_VAL >= 1:
        spf[0] = 0
    if MAX_VAL >= 2:
        spf[1] = 1
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Step 2: Compute Powerful Part P(x) for all x
    # P(x) is the product of p^e where e >= 2
    P = [1] * (MAX_VAL + 1)
    
    for i in range(2, MAX_VAL + 1):
        x = i
        p_part = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            if exp >= 2:
                p_part *= (p ** exp)
        P[i] = p_part
        
        # Check deadline
        if i % 10000 == 0 and time.time() - start_time > deadline - 5:
            # If we run out of time, we break. 
            # However, with SPF, this loop is very fast (~1-2s for 1e6).
            # This check is just a safety net.
            pass

    # Step 3: Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = 0
    
    # We can optimize the search slightly by checking time, but the loop is trivial (1e6 iterations)
    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() - start_time > deadline - 2:
            break

    return best_n
# EVOLVE-BLOCK-END
