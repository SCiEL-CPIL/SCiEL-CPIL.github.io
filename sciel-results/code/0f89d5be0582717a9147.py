import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 75.0
    
    N = 10_000_000
    
    # Step 1: Sieve sigma(n) for n=1..N using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return [11, 6]
    
    # Step 2: Search for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let num = |2a - 3b|. We want to minimize num / (2b).
    # Iterate b from N down to 1.
    # If 1/(2b) > best_error, no smaller b can improve (since num >= 1).
    
    best_num = 10**18
    best_den = 1
    best_a, best_b = 11, 6 # fallback
    
    for b in range(N, 0, -1):
        if time.time() - start_time > DEADLINE - 2:
            break
            
        # Pruning: if even the best possible error for this b (num=1) is worse than current best, break.
        # Best possible error is 1 / (2b).
        # If 1 / (2b) > best_num / best_den, then break.
        # <=> best_den > best_num * 2 * b
        # <=> best_den > 2 * b * best_num
        
        if best_den > 2 * b * best_num:
            break
            
        b_int = b
        sigma_b = sigma[b_int]
        
        # Determine candidate a's
        # 2a - 3b = k
        # a = (3b + k) / 2
        # k must have same parity as 3b (so 3b+k is even).
        # If b is odd, 3b is odd, so k must be odd. Min |k| is 1.
        # If b is even, 3b is even, so k must be even. Min |k| is 2 (k=0 implies a=1.5b, not int if b odd? No, if b even, 1.5b is int. k=0 => a=1.5b. Error 0. But task says raw value 0 is invalid. So we must avoid k=0).
        
        candidates = []
        if b_int % 2 == 1:
            # b is odd, 3b is odd. k must be odd.
            # k = 1 => a = (3b+1)/2
            # k = -1 => a = (3b-1)/2
            a1 = (3 * b_int + 1) // 2
            a2 = (3 * b_int - 1) // 2
            candidates = [a1, a2]
        else:
            # b is even, 3b is even. k must be even.
            # k = 2 => a = (3b+2)/2 = 1.5b + 1
            # k = -2 => a = (3b-2)/2 = 1.5b - 1
            # k = 0 => a = 1.5b (Invalid, error 0)
            a1 = (3 * b_int + 2) // 2
            a2 = (3 * b_int - 2) // 2
            candidates = [a1, a2]
            
        for a in candidates:
            if a < 1 or a > N:
                continue
            if sigma[a] == sigma_b:
                # Calculate error
                # num = |2a - 3b|
                num = abs(2 * a - 3 * b_int)
                if num == 0:
                    continue
                den = 2 * b_int
                # Compare num/den < best_num/best_den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a, b_int
    
    # Ensure Python int types
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]
