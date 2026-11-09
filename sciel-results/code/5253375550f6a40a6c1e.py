import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    M = 100000000  # 10^8
    
    P = generate_powerful(M)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    size = 2 * M + 1
    A = np.zeros(size, dtype=np.float32)
    for x in P:
        A[x] = 1.0
        
    C = np.convolve(A, A, mode='full')
    
    # C is float32, values are integers.
    # We only care about indices 0 to 2*M
    C_int = np.rint(C[:size]).astype(np.int64)
    
    delta = np.zeros(size, dtype=np.int64)
    # delta[2*i] = 1 if A[i] == 1
    # A is float32
    delta[0:size:2] = (A[0:M+1] > 0.5).astype(np.int64)
    
    D = (C_int + delta) // 2
    
    max_count = np.max(D[2:])
    if max_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = np.argmax(D[2:]) + 2
    
    P_set = set(P)
    pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
