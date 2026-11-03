import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # 1. Generate Legendre sequence
    p = 401
    a = np.zeros(N, dtype=np.int8)
    for i in range(N):
        val = pow(i + 1, (p - 1) // 2, p)
        a[i] = 1 if val == 1 else -1

    # Helper to compute ACF
    def get_acf(a):
        # Returns R[1]...R[N-1]
        res = np.zeros(N - 1, dtype=np.int16)
        for k in range(1, N):
            res[k - 1] = np.dot(a[:N - k], a[k:])
        return res

    acf = get_acf(a)
    current_max = np.max(np.abs(acf))
    
    # Greedy Local Search
    # We iterate through indices. If flipping reduces max, accept.
    # We do multiple passes until no improvement or time limit.
    
    while (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
                
            old_val = a[i]
            a[i] = -old_val
            
            # Update ACF
            # Delta R[k] = -2 * old_val * (a[i+k] + a[i-k])
            # k ranges 1 to max(i, N-1-i)
            max_k = max(i, N - 1 - i)
            for k in range(1, max_k + 1):
                delta = 0
                if i <= N - 1 - k:
                    delta += a[i + k]
                if i - k >= 0:
                    delta += a[i - k]
                
                if delta != 0:
                    acf[k - 1] += -2 * old_val * delta
            
            new_max = np.max(np.abs(acf))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                a[i] = old_val
                for k in range(1, max_k + 1):
                    delta = 0
                    if i <= N - 1 - k:
                        delta += a[i + k]
                    if i - k >= 0:
                        delta += a[i - k]
                    if delta != 0:
                        acf[k - 1] -= -2 * old_val * delta
        
        if not improved:
            break

    return a
