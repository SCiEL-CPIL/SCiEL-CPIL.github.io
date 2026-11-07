import numpy as np
import time
import sys

def construct():
    # Deadline guard: 80 seconds max
    start_time = time.time()
    deadline = start_time + 80
    
    N = 10000000
    target_ratio_num = 3
    target_ratio_den = 2
    
    # 1. Compute sigma(n) for all n up to N using NumPy sieve
    sigma = np.ones(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 10:
        return [11, 6]

    # Convert to list for faster access in the search loop
    sigma_list = sigma.tolist()
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    max_b = int(N * 2 / 3)
    
    # Iterate b from large to small
    # If 1/(2b) >= best_error, we can break
    for b in range(max_b, 0, -1):
        # Pruning condition: even the best possible a (k=1 or 2) gives error >= best_error
        # Min error for this b is approx 1/(2b)
        if 1.0 / (2.0 * b) >= best_error:
            break
            
        target_a_float = 1.5 * b
        # Check integers around target_a_float
        # Widen window to +/- 10 to find larger b solutions with slightly larger k
        a_start = max(1, int(target_a_float - 10))
        a_end = min(N, int(target_a_float + 10))
        
        sb = sigma_list[b]
        
        for a in range(a_start, a_end + 1):
            if sigma_list[a] == sb:
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue
                err_den = 2 * b
                error = err_num / err_den
                
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                    
        # Time check
        if time.time() > deadline - 5:
            break

    # Ensure types are Python ints
    return [int(best_a), int(best_b)]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}")
    else:
        print("Type error")
