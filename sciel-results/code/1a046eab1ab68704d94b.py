import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 4]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, T):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - T[i,j,k]) > 1e-10:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, -1, -1, -1, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 20s margin
    
    T = _build_target_tensor()
    
    # Try structured decomposition with Strassen
    # i in {0,1}: 4 blocks of 2x2x2, each rank 7 -> 28
    # i = 2: 1x4x4, rank 1
    # Total: 29
    
    U, V, W = _strassen_2x2x2()
    
    # Build full decomposition
    # For i_offset in {0, 1}, j_offset in {0, 2}, k_offset in {0, 2}:
    #   Use Strassen for the 2x2x2 block
    
    rank_total = 4 * 7 + 1  # 29
    U_full = np.zeros((n*m, rank_total))
    V_full = np.zeros((m*p, rank_total))
    W_full = np.zeros((p*n, rank_total))
    
    col = 0
    for i_off in [0, 1]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                for r in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            i_idx = i_off + ii
                            j_idx = j_off + jj
                            U_full[i_idx*m + j_idx, col] = U[ii*2+jj, r]
                    for jj in range(2):
                        for kk in range(2):
                            j_idx = j_off + jj
                            k_idx = k_off + kk
                            V_full[j_idx*p + k_idx, col] = V[jj*2+kk, r]
                    for kk in range(2):
                        for ii in range(2):
                            k_idx = k_off + kk
                            i_idx = i_off + ii
                            W_full[k_idx*n + i_idx, col] = W[kk*2+ii, r]
                    col += 1
    
    # For i=2: rank 1 term
    # T[2,j,k] = 1 for all j,k
    # U[2*m+j, col] = 1 for all j
    # V[j*p+k, col] = 1 for all j,k
    # W[k*n+2, col] = 1 for all k
    for j in range(m):
        U_full[2*m + j, col] = 1
    for j in range(m):
        for k in range(p):
            V_full[j*p + k, col] = 1
    for k in range(p):
        W_full[k*n + 2, col] = 1
    col += 1
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, T):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p  # 3*4*4 = 48
    U_school = np.zeros((n*m, rank))
    V_school = np.zeros((m*p, rank))
    W_school = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
