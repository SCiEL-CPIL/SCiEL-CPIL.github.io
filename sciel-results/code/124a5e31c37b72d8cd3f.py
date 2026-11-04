"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b00)
    # M4 = a11 * (b11 - b01)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],   # a01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # a10: M2, M6
        [1, 1, 0, 1, 0, 0, -1],  # a11: M1, M2, M4, M7
    ], dtype=float)
    
    V = np.array([
        [1, 1, -1, 0, 0, 1, 0],  # b00: M1, M2, M3, M6
        [0, 0, 0, -1, 0, 1, 0],  # b01: M4, M6
        [0, 0, 1, 0, 0, 0, 1],   # b10: M3, M7
        [1, 0, 0, 1, 1, 0, 1],   # b11: M1, M4, M5, M7
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0,i=0): M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=0,i=1): M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1,i=0): M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # C11 (k=1,i=1): M1, -M2, M3, M6
    ], dtype=float)
    
    # Verify
    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(7):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4x4 using Strassen's algorithm.
    Split into 2x2 blocks and use the tensor product of Strassen decompositions.
    """
    n, m, p = 4, 4, 4
    
    # Get the 2x2x2 Strassen decomposition
    base = _strassen_2x2x2()
    if base is None:
        return None
    
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    rank = 49  # 7 * 7
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The 4x4x4 matrix multiplication tensor is the tensor product of two 2x2x2 tensors.
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # c = 7*c1 + c2
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            
                            for c1 in range(7):
                                for c2 in range(7):
                                    c = 7 * c1 + c2
                                    # U[i, c] = U_base[i1, c1] * U_base[i2, c2]
                                    U[i * m + j, c] += U_base[i1 * 2 + j1, c1] * U_base[i2 * 2 + j2, c2]
                                    # V[j, c] = V_base[j1, c1] * V_base[j2, c2]
                                    V[j * p + k, c] += V_base[j1 * 2 + k1, c1] * V_base[j2 * 2 + k2, c2]
                                    # W[k, c] = W_base[k1, c1] * W_base[k2, c2]
                                    W[k * n + i, c] += W_base[k1 * 2 + i1, c1] * W_base[k2 * 2 + i2, c2]
    
    # Verify
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x5_strassen():
    """
    Construct a decomposition for 4x4x5 using Strassen-based approach.
    
    Strategy: Use Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    Total rank: 49 + 16 = 65.
    """
    n, m, p = 4, 4, 5
    
    # For the 4x4x4 part, use the rank-49 Strassen decomposition
    # For the 4x4x1 part, use schoolbook (rank 16)
    
    result_4x4x4 = _construct_4x4x4_strassen()
    if result_4x4x4 is None:
        return _schoolbook(n, m, p)
    
    U4 = np.array(result_4x4x4["U"], dtype=float)
    V4 = np.array(result_4x4x4["V"], dtype=float)
    W4 = np.array(result_4x4x4["W"], dtype=float)
    
    rank4 = U4.shape[1]  # 49
    rank_total = rank4 + 16
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Copy the 4x4x4 part (k=0..3)
    # U[i*m+j, c] for c in 0..48
    U[:, :rank4] = U4
    
    # V[j*p+k, c] for c in 0..48, k in 0..3
    for j in range(m):
        for k in range(4):
            V[j * p + k, :rank4] = V4[j * 4 + k, :rank4]
    
    # W[k*n+i, c] for c in 0..48, k in 0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, :rank4] = W4[k * n + i, :rank4]
    
    # Add the 4x4x1 part (k=4) using schoolbook
    # For each (i, j), add a term for k=4
    col = rank4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # Verify
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave margin before 150s kill
    
    # Try the Strassen-based construction
    try:
        result = _construct_4x4x5_strassen()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
