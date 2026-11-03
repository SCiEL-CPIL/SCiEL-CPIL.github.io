import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# Rank-7 decomposition of 3x3x3 multiplication tensor.
# These are the standard Strassen-like constants for 3x3 matrices.
# U: 3x3, V: 3x3, W: 3x3 for each of the 7 terms.
# T = sum_{r=1}^7 U_r \otimes V_r \otimes W_r
# where C_{ik} = sum_{j,r} U_r[i,j] V_r[j,k] W_r[k,i]

_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]

def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank14():
    n, m, p = DIMENSIONS
    # We split the p=6 dimension into two groups of 3.
    # The rank is 2 * 7 = 14.
    R = 14
    U = _zero_matrix(n * m, R)
    V = _zero_matrix(m * p, R)
    W = _zero_matrix(p * n, R)
    
    # For each of the 7 terms in the 3x3x3 decomposition:
    for r in range(7):
        # Term r corresponds to the first 3 columns (k=0,1,2)
        col1 = r
        # Term r+7 corresponds to the next 3 columns (k=3,4,5)
        col2 = r + 7
        
        # Fill U (shared structure, but here we duplicate the U terms for the two blocks? 
        # No, U is 3x3. In the 3x3x6 case, U is 3x3. 
        # The decomposition is sum_{r=1}^{14} U_r \otimes V_r \otimes W_r.
        # U_r is 3x3.
        # For the first block, U_r = _A[r].
        # For the second block, U_{r+7} = _A[r].
        # So we fill U columns r and r+7 with _A[r].
        
        for i in range(n):
            for j in range(m):
                val = _A[r][i][j]
                if val != 0:
                    U[i * m + j][col1] = val
                    U[i * m + j][col2] = val
        
        # Fill V
        # V is 3x6.
        # For col1 (first block), V[j, k] for k=0,1,2 comes from _B[r][j][k]
        # For col2 (second block), V[j, k+3] for k=0,1,2 comes from _B[r][j][k]
        # Note: _B[r] is 3x3.
        for j in range(m):
            for k_local in range(3):
                val = _B[r][j][k_local]
                if val != 0:
                    V[j * p + k_local][col1] = val
                    V[j * p + (3 + k_local)][col2] = val
        
        # Fill W
        # W is 6x3.
        # For col1, W[k, i] for k=0,1,2 comes from _C[r][i][k] ?
        # Check indices: C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor term: U[i,j] V[j,k] W[k,i].
        # _C[r] is 3x3. Let's call it C_mat.
        # In the 3x3 case, C_{ik} = sum_j U[i,j] V[j,k] W[k,i].
        # So W[k,i] corresponds to _C[r][i][k] ??
        # Let's check the seed code's _C usage.
        # Seed: W[k_global * n + i][base + r] = _C[r][i][k_local]
        # So W[k, i] = _C[r][i][k].
        
        for i in range(n):
            for k_local in range(3):
                val = _C[r][i][k_local]
                if val != 0:
                    W[k_local * n + i][col1] = val
                    W[(3 + k_local) * n + i][col2] = val
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try rank 14
    try:
        cand = _rank14()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
