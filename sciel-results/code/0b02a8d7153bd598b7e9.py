import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80

    MAXV = 1000001

    # Use int32 for memory efficiency and potential speedup
    lpf = np.zeros(MAXV + 1, dtype=np.int32)
    
    is_prime = np.ones(MAXV + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(MAXV)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    for p in primes:
        lpf[p::p] = p

    n_arr = np.arange(10000, 1000001, dtype=np.int32)
    lpf_n = lpf[n_arr]
    lpf_np1 = lpf[n_arr + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)

    min_idx = np.argmin(max_lpf)
    best_n = int(n_arr[min_idx])

    if time.time() > deadline:
        pass

    return best_n
