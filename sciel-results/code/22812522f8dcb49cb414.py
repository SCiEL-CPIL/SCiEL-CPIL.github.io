import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 60.0
    max_time_allowed = deadline - 5.0

    N = 1000000
    LOWER = 10000
    UPPER = N

    # Step 1: SPF
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Step 2: Powerful Part
    pow_part = [1] * (N + 2)
    for i in range(2, N + 1):
        p = spf[i]
        m = i
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        if e >= 2:
            pow_part[i] = pow_part[m] * (p ** e)
        else:
            pow_part[i] = pow_part[m]

    if time.time() - start_time > max_time_allowed:
        return 10000

    # Step 3: Find max using NumPy
    # Convert to numpy array of int64
    # Max P(n) is 10^6, product is 10^12, fits in int64
    P_np = np.array(pow_part, dtype=np.int64)
    
    # We need P[n] * P[n+1] for n in [LOWER, N-1]
    # P_np[LOWER:N] corresponds to P[LOWER] ... P[N-1]
    # P_np[LOWER+1:N+1] corresponds to P[LOWER+1] ... P[N]
    scores = P_np[LOWER:N] * P_np[LOWER+1:N+1]
    
    # Find index of max score
    max_idx = np.argmax(scores)
    best_n = LOWER + max_idx
    
    return best_n
