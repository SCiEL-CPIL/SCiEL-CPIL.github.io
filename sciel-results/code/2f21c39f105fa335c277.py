import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, keep 5s margin
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N
    # sigma[i::i] += i is the standard sieve for sigma
    # This takes roughly 10-15 seconds for N=10^7
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 10:
        # Fallback if sigma computation took too long (unlikely)
        return [11, 6]

    # 2. Search for best pair
    # We iterate b downwards.
    # If we find a solution with k=1 (i.e., |2a-3b|=1), it is the global optimum.
    # We can return immediately.
    # If we don't find k=1, we keep the best solution found (min |2a-3b|/b).
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We only need to check a few small k values.
    # k=1 is the best possible.
    # k=2 is the next best for even b.
    # k=3 is the next best for odd b (if k=1 fails).
    
    for b in range(N, 0, -1):
        sb = sigma[b]
        
        # Check k=1
        # 2a = 3b + 1 => a = (3b+1)/2
        # 2a = 3b - 1 => a = (3b-1)/2
        # These are integers if b is odd.
        if b % 2 == 1:
            # a1 = (3b + 1) // 2
            a1 = (3 * b + 1) // 2
            if a1 <= N:
                if sigma[a1] == sb:
                    return [int(a1), int(b)]
            
            # a2 = (3b - 1) // 2
            a2 = (3 * b - 1) // 2
            if a2 >= 1 and a2 <= N:
                if sigma[a2] == sb:
                    return [int(a2), int(b)]
            
            # Check k=3 as fallback (only if we haven't found k=1 yet, which we haven't since we didn't return)
            # a3 = (3b + 3) // 2
            a3 = (3 * b + 3) // 2
            if a3 <= N:
                if sigma[a3] == sb:
                    err = 3.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a3
                        best_b = b
            
            a4 = (3 * b - 3) // 2
            if a4 >= 1 and a4 <= N:
                if sigma[a4] == sb:
                    err = 3.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a4
                        best_b = b
        else:
            # b is even
            # k=1 is not possible (2a-3b=odd, but 2a is even, 3b is even, diff is even)
            # Check k=2
            # 2a = 3b + 2 => a = (3b+2)/2
            a1 = (3 * b + 2) // 2
            if a1 <= N:
                if sigma[a1] == sb:
                    err = 2.0 / (2 * b) # 1/b
                    if err < best_error:
                        best_error = err
                        best_a = a1
                        best_b = b
                        # Note: Cannot return here. A k=1 solution with smaller b might be better.
                        # But we are iterating downwards. If we found k=1, we would have returned.
                        # So no k=1 solution exists for b' > b.
                        # Could a k=1 solution at b' < b be better?
                        # 1/(2b') < 1/b => b < 2b'. Possible.
                        # So we must continue.
            
            a2 = (3 * b - 2) // 2
            if a2 >= 1 and a2 <= N:
                if sigma[a2] == sb:
                    err = 1.0 / b
                    if err < best_error:
                        best_error = err
                        best_a = a2
                        best_b = b

            # Check k=4 as fallback
            a3 = (3 * b + 4) // 2
            if a3 <= N:
                if sigma[a3] == sb:
                    err = 4.0 / (2 * b) # 2/b
                    if err < best_error:
                        best_error = err
                        best_a = a3
                        best_b = b
            
            a4 = (3 * b - 4) // 2
            if a4 >= 1 and a4 <= N:
                if sigma[a4] == sb:
                    err = 2.0 / b
                    if err < best_error:
                        best_error = err
                        best_a = a4
                        best_b = b

        # Time check
        if b % 1000000 == 0:
            if time.time() > deadline - 5:
                break

    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    res = construct()
    print(res)
