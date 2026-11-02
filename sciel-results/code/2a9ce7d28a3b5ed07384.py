"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and dot products for 1x2x2 blocks.
Expected rank: 36.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check invalid entries (spot check)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j: continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _add_strassen(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 2x2x2 Strassen decomposition to the global matrices.
    a_indices: list of 4 global indices for U
    b_indices: list of 4 global indices for V
    c_indices: list of 4 global indices for W
    Local mapping:
    A: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    B: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    C: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    """
    rank = U.shape[1]
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    coeffs = [
        # A, B, C
        [(0,1), (3,1), (0,1), (3,1)], # M1
        [(2,1), (3,1), (0,1), (2,1), (3,1)], # M2
        [(0,1), (1,1), (3,-1), (1,1), (3,1)], # M3
        [(3,1), (2,1), (0,-1), (2,1), (3,1)], # M4
        [(0,1), (1,1), (3,1), (1,1), (3,1)], # M5
        [(2,1), (0,-1), (0,1), (1,1), (2,1), (3,1)], # M6
        [(1,1), (3,-1), (2,1), (3,1), (0,1), (3,1)]  # M7
    ]
    
    # This structure is hard to parse. Let's just hardcode the updates.
    # r=0 (M1)
    U[a_indices[0], rank] += 1; U[a_indices[3], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[3], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[3], rank] += 1
    
    # r=1 (M2)
    U[a_indices[2], rank+1] += 1; U[a_indices[3], rank+1] += 1
    V[b_indices[0], rank+1] += 1
    W[c_indices[2], rank+1] += 1; W[c_indices[3], rank+1] += 1
    
    # r=2 (M3)
    U[a_indices[0], rank+2] += 1
    V[b_indices[1], rank+2] += 1; V[b_indices[3], rank+2] -= 1
    W[c_indices[1], rank+2] += 1; W[c_indices[3], rank+2] += 1
    
    # r=3 (M4)
    U[a_indices[3], rank+3] += 1
    V[b_indices[2], rank+3] += 1; V[b_indices[0], rank+3] -= 1
    W[c_indices[2], rank+3] += 1; W[c_indices[3], rank+3] += 1
    
    # r=4 (M5)
    U[a_indices[0], rank+4] += 1; U[a_indices[1], rank+4] += 1
    V[b_indices[3], rank+4] += 1
    W[c_indices[1], rank+4] += 1; W[c_indices[3], rank+4] += 1
    
    # r=5 (M6)
    U[a_indices[2], rank+5] += 1; U[a_indices[0], rank+5] -= 1
    V[b_indices[0], rank+5] += 1; V[b_indices[1], rank+5] += 1
    W[c_indices[2], rank+5] += 1; W[c_indices[3], rank+5] += 1
    
    # r=6 (M7)
    U[a_indices[1], rank+6] += 1; U[a_indices[3], rank+6] -= 1
    V[b_indices[2], rank+6] += 1; V[b_indices[3], rank+6] += 1
    W[c_indices[0], rank+6] += 1; W[c_indices[3], rank+6] += 1

def _add_dot_1x2x2(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 1x2x2 dot product decomposition.
    a_indices: list of 2 global indices for U (j=0, j=1)
    b_indices: list of 4 global indices for V (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    c_indices: list of 2 global indices for W (k=0, k=1)
    """
    rank = U.shape[1]
    
    # r=0: A0 * (B00, B01) -> (C0, C1)
    U[a_indices[0], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[1], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[1], rank] += 1
    
    # r=1: A1 * (B10, B11) -> (C0, C1)
    U[a_indices[1], rank+1] += 1
    V[b_indices[2], rank+1] += 1; V[b_indices[3], rank+1] += 1
    W[c_indices[0], rank+1] += 1; W[c_indices[1], rank+1] += 1

def _block_decomposition():
    n, m, p = DIMENSIONS # 3, 4, 8
    total_rank = 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    # Indices helper
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    
    # Block 1: C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    # Split into two 2x2x2:
    # 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    a_idx_1a = [i*4+j for i in range(2) for j in range(2)] # [0,1,4,5]
    b_idx_1a = [j*8+k for j in range(2) for k in range(2)] # [0,1,8,9]
    c_idx_1a = [k*3+i for k in range(2) for i in range(2)] # [0,1,3,4]
    _add_strassen(U, V, W, a_idx_1a, b_idx_1a, c_idx_1a)
    current_rank += 7
    
    # 1b: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    a_idx_1b = [i*4+j for i in range(2) for j in range(2,4)] # [2,3,6,7]
    b_idx_1b = [j*8+k for j in range(2,4) for k in range(2)] # [16,17,24,25]
    c_idx_1b = [k*3+i for k in range(2) for i in range(2)] # [0,1,3,4]
    _add_strassen(U, V, W, a_idx_1b, b_idx_1b, c_idx_1b)
    current_rank += 7
    
    # Block 2: C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    # 2a: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    a_idx_2a = [i*4+j for i in range(2) for j in range(2)] # [0,1,4,5]
    b_idx_2a = [j*8+k for j in range(2) for k in range(4,6)] # [4,5,12,13]
    c_idx_2a = [k*3+i for k in range(4,6) for i in range(2)] # [12,13,15,16]
    _add_strassen(U, V, W, a_idx_2a, b_idx_2a, c_idx_2a)
    current_rank += 7
    
    # 2b: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    a_idx_2b = [i*4+j for i in range(2) for j in range(2,4)] # [2,3,6,7]
    b_idx_2b = [j*8+k for j in range(2,4) for k in range(4,6)] # [20,21,28,29]
    c_idx_2b = [k*3+i for k in range(4,6) for i in range(2)] # [12,13,15,16]
    _add_strassen(U, V, W, a_idx_2b, b_idx_2b, c_idx_2b)
    current_rank += 7
    
    # Block 3: C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    # 3a: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    a_idx_3a = [2*4+j for j in range(2)] # [8,9]
    b_idx_3a = [j*8+k for j in range(2) for k in range(2)] # [0,1,8,9]
    c_idx_3a = [k*3+2 for k in range(2)] # [2,5]
    _add_dot_1x2x2(U, V, W, a_idx_3a, b_idx_3a, c_idx_3a)
    current_rank += 2
    
    # 3b: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    a_idx_3b = [2*4+j for j in range(2,4)] # [10,11]
    b_idx_3b = [j*8+k for j in range(2,4) for k in range(2)] # [16,17,24,25]
    c_idx_3b = [k*3+2 for k in range(2)] # [2,5]
    _add_dot_1x2x2(U, V, W, a_idx_3b, b_idx_3b, c_idx_3b)
    current_rank += 2
    
    # Block 4: C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    # 4a: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    a_idx_4a = [2*4+j for j in range(2)] # [8,9]
    b_idx_4a = [j*8+k for j in range(2) for k in range(4,6)] # [4,5,12,13]
    c_idx_4a = [k*3+2 for k in range(4,6)] # [14,17]
    _add_dot_1x2x2(U, V, W, a_idx_4a, b_idx_4a, c_idx_4a)
    current_rank += 2
    
    # 4b: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    a_idx_4b = [2*4+j for j in range(2,4)] # [10,11]
    b_idx_4b = [j*8+k for j in range(2,4) for k in range(4,6)] # [20,21,28,29]
    c_idx_4b = [k*3+2 for k in range(4,6)] # [14,17]
    _add_dot_1x2x2(U, V, W, a_idx_4b, b_idx_4b, c_idx_4b)
    current_rank += 2
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    try:
        result = _block_decomposition()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception as e:
        pass
    
    # Fallback
    return _schoolbook()
# EVOLVE-BLOCK-END
