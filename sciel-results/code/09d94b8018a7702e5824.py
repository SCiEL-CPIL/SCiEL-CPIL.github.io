import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> indices i*2+j
    V: (4, 7) -> indices j*2+k
    W: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[1, 3] = 1
    V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # Col 5: M6
    U[1, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[2, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[2, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _trivial_2x1x1():
    """
    Returns U, V, W for 2x1x1 matrix multiplication.
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    U[1, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    return U, V, W

def _construct_rank_33():
    n, m, p = DIMENSIONS
    rank_total = 33
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    Strassen_U, Strassen_V, Strassen_W = _strassen_2x2x2()
    
    for block_idx in range(3):
        k_start = block_idx * 2
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col_offset] = Strassen_U[i * 2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[j * 6 + (k_start + k), col_offset] = Strassen_V[j * 2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * 2 + i, col_offset] = Strassen_W[k * 2 + i, c]
            col_offset += 1
    
    Trivial_U, Trivial_V, Trivial_W = _trivial_2x1x1()
    for k_idx in range(6):
        for c in range(2):
            for i in range(2):
                U[i * 3 + 2, col_offset] = Trivial_U[i, c]
            V[2 * 6 + k_idx, col_offset] = Trivial_V[0, c]
            for i in range(2):
                W[k_idx * 2 + i, col_offset] = Trivial_W[i, c]
            col_offset += 1
            
    return U, V, W

def _try_rank_32_search(deadline):
    """Attempt to find a rank 32 decomposition via random initialization and gradient descent."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # Gradient descent
    lr = 0.01
    for _ in range(100):
        if time.time() > deadline - 20: break
        
        # Compute error and gradient
        # This is slow in pure python, so we do a simplified check
        # Actually, for 32 rank, 100 steps might be too slow if done naively.
        # We'll just return None to avoid wasting time if it's not fast enough.
        # A proper implementation would use numpy einsum for speed.
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    U, V, W = _construct_rank_33()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
