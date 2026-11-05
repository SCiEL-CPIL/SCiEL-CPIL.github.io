"""Rewritten program using modular composition for matrix multiplication tensor decomposition."""
import numpy as np

def _trivial_decomp(n, m, p):
    """
    Returns U, V, W for trivial decomposition of n x m * m x p -> n x p.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 3, 3, 5
    
    # Split p into p1 and p2
    p1 = 3
    p2 = 2
    
    # Part 1: n x m * m x p1
    U1, V1, W1 = _trivial_decomp(n, m, p1)
    
    # Part 2: n x m * m x p2
    U2, V2, W2 = _trivial_decomp(n, m, p2)
    
    # Combine
    # U_total: (n*m, rank1 + rank2)
    U_total = np.hstack([U1, U2])
    
    # V_total: (m*p, rank1 + rank2)
    # V1 corresponds to B[:, :p1]
    # V2 corresponds to B[:, p1:]
    # V_total is block diagonal in the input dimension?
    # V is (m*p, Rank).
    # V1 is (m*p1, Rank1).
    # V2 is (m*p2, Rank2).
    # We need to embed V1 into the first m*p1 rows of V_total.
    # And V2 into the last m*p2 rows.
    V_total = np.zeros((m * p, U1.shape[1] + U2.shape[1]), dtype=np.float64)
    V_total[:m*p1, :U1.shape[1]] = V1
    V_total[m*p1:, U1.shape[1]:] = V2
    
    # W_total: (p*n, rank1 + rank2)
    # W1 corresponds to C[:, :p1]
    # W2 corresponds to C[:, p1:]
    # W is (p*n, Rank).
    # W1 is (p1*n, Rank1).
    # W2 is (p2*n, Rank2).
    # Embed W1 into first p1*n rows.
    # Embed W2 into last p2*n rows.
    W_total = np.zeros((p * n, U1.shape[1] + U2.shape[1]), dtype=np.float64)
    W_total[:p1*n, :U1.shape[1]] = W1
    W_total[p1*n:, U1.shape[1]:] = W2
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
