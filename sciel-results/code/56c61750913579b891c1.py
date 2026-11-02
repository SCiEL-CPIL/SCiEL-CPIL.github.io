"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the constant tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T_recon, T_expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_decompose(n, m, p):
    """
    Block decomposition: split 3x4 into [1x4; 2x4] and 4x6 into [4x2, 4x2, 4x2].
    Top row: 1x4 * 4x2 = 1x2 (4 mults each, 3 blocks = 12)
    Bottom: 2x4 * 4x2 = 2x2 (16 mults each schoolbook, 3 blocks = 48)
    Total: 60
    """
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * rank for _ in range(p * n)]  # 18 x 60
    
    col = 0
    
    # Top row: i=0, for each block b (0,1,2)
    for b in range(3):
        for j in range(4):
            for k in range(2):
                # U[0*m + j, col] = 1
                # V[j*p + (2*b + k), col] = 1
                # W[(2*b + k)*n + 0, col] = 1
                U[0 * m + j][col] = 1
                V[j * p + (2 * b + k)][col] = 1
                W[(2 * b + k) * n + 0][col] = 1
                col += 1
    
    # Bottom rows: i=1,2, for each block b (0,1,2)
    for b in range(3):
        for i in range(1, 3):  # i=1,2
            for j in range(4):
                for k in range(2):
                    # U[i*m + j, col] = 1
                    # V[j*p + (2*b + k), col] = 1
                    # W[(2*b + k)*n + i, col] = 1
                    U[i * m + j][col] = 1
                    V[j * p + (2 * b + k)][col] = 1
                    W[(2 * b + k) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try block decomposition (rank 60)
    result = _block_decompose(n, m, p)
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

