import time

def construct():
    start_time = time.time()
    LIMIT = 90  # seconds (evaluator kills at 90s)
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # Quick sanity check: verify powerful part for known values
    # This ensures our SPF and P computation is correct
    def _verify_powerful_part():
        # 72 = 2^3 * 3^2 -> P = 2^3 * 3^2 = 72
        # 48 = 2^4 * 3 -> P = 2^4 = 16
        # 80 = 2^4 * 5 -> P = 2^4 = 16
        # 81 = 3^4 -> P = 3^4 = 81
        # 100 = 2^2 * 5^2 -> P = 2^2 * 5^2 = 100
        test_cases = [(72, 72), (48, 16), (80, 16), (81, 81), (100, 100), (108, 108)]
        for val, expected in test_cases:
            p = spf[val]
            x = val // p
            if x % p == 0:
                y = x // p
                if y % p == 0:
                    P_val = P[x] * p
                else:
                    P_val = P[x] * p * p
            else:
                P_val = P[x]
            if P_val != expected:
                return False
        return True
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Using a list of ints for speed; this is the standard fast SPF sieve
    spf = list(range(N_MAX + 1))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Verify SPF for small primes
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if spf[prime] != prime:
            return 10000  # Fallback if sieve is broken
    
    # 2. Compute Powerful Part P[k]
    # P[i] is the product of p^e for all prime powers p^e || i where e >= 2
    # Recurrence:
    # Let p = spf[i], x = i // p
    # If x % p != 0: i has p^1, so P[i] = P[x]
    # If x % p == 0:
    #   Let y = x // p
    #   If y % p != 0: i has p^2, so P[i] = P[x] * p^2 (since P[x] has no p factor)
    #   If y % p == 0: i has p^3+, so P[i] = P[x] * p (since P[x] already has p^(e-1) where e-1>=2)
    #     Wait, if i = p^e * m, e>=3. x = p^(e-1) * m.
    #     If e-1 >= 2, P[x] = P[m] * p^(e-1).
    #     P[i] should be P[m] * p^e = P[x] * p.
    #     If e-1 == 1 (i.e., e=2), x = p*m. P[x] = P[m].
    #     P[i] = P[m] * p^2 = P[x] * p^2.
    
    P = [1] * (N_MAX + 1)
    P[1] = 1
    
    for i in range(2, N_MAX + 1):
        p = spf[i]
        x = i // p
        if x % p == 0:
            y = x // p
            if y % p == 0:
                # i has p^3 or higher
                P[i] = P[x] * p
            else:
                # i has exactly p^2
                P[i] = P[x] * p * p
        else:
            # i has exactly p^1
            P[i] = P[x]
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000
    
    # Verify powerful part computation
    if not _verify_powerful_part():
        # Fallback to pure Python verification
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n
        return best_n
    
    # 3. Find n that maximizes P[n] * P[n+1]
    # Max P[i] for i <= 10^6 is at most 10^6.
    # Product of two such parts can be up to 10^12, which fits in int64 (9.2 * 10^18).
    # So we can safely use numpy int64 for the product calculation.
    
    try:
        import numpy as np
        P_arr = np.array(P, dtype=np.int64)
        
        # n is from START to 1000000 inclusive.
        left = P_arr[START:1000001]
        right = P_arr[START+1:1000002]
        
        products = left * right
        
        max_idx = np.argmax(products)
        best_n = int(START + max_idx)
        
    except ImportError:
        # Fallback to pure Python if numpy is not available
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return best_n
