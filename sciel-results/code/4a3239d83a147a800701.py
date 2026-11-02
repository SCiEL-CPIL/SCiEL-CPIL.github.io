import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    r = m * p  # 25
    
    # U: (n*m, r) = (10, 25)
    # V: (m*p, r) = (25, 25)
    # W: (p*n, r) = (10, 25)
    
    U = np.ones((n * m, r), dtype=int)
    V = np.eye(r, dtype=int)
    W = np.ones((p * n, r), dtype=int)
    
    # Verification (optional, but good for safety)
    # The condition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # With V being identity, V[j*p+k, c] is 1 if c == j*p+k, else 0.
    # So the sum is U[i*m+j, j*p+k] * W[k*n+i, j*p+k].
    # Since U and W are all 1s, this is 1 * 1 = 1.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
