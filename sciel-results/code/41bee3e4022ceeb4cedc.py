import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Use standard rank-100 decomposition as guaranteed valid baseline
    rank = n * m * p  # 100
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify correctness
    is_valid = True
    for test_i in range(n):
        for test_j in range(m):
            for test_k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[test_i * m + test_j, r] * V[test_j * p + test_k, r] * W[test_k * n + test_i, r]
                if actual != expected:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # This should never happen for standard decomposition
        raise ValueError("Standard decomposition failed verification")
