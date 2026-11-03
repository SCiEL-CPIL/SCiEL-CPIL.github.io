import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_333():
    """
    Placeholder - not used in construct().
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 32 decomposition
    result_32 = _strassen_32()
    ok, err = _verify_fast(result_32["U"], result_32["V"], result_32["W"], n, m, p)
    
    if ok:
        return result_32
    
    # Fallback to schoolbook
    return _schoolbook()

def _strassen_32():
    """
    Rank 32 decomposition for 3x4x4 matrix multiplication.
    
    Strategy:
    - Rows 0,1 of A (2x4) times B (4x4) -> C rows 0,1 (2x4): use 4 blocks of 2x2x2 Strassen = 28 terms
    - Row 2 of A (1x4) times B (4x4) -> C row 2 (1x4): rank 4 (one per output column)
    Total: 28 + 4 = 32
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term, we have (U_coeffs, V_coeffs, W_coeffs)
    # U: 2x2 matrix (rows i=0,1; cols j=0,1)
    # V: 2x2 matrix (rows j=0,1; cols k=0,1)
    # W: 2x2 matrix (rows k=0,1; cols i=0,1) -- note: W[k*n+i, r]
    
    strassen_terms = [
        # (U, V, W) as dict {(i,j): val}
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,0): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 4 blocks: A row-block (2 rows), A col-block, B col-block, C col-block
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x4: rows 0-3, cols 0-3
    # C is 2x4: rows 0-1, cols 0-3
    #
    # Block (ac, bc): A rows 0-1, cols ac*2..ac*2+1; B rows bc*2..bc*2+1, cols 0-3
    # Wait, I need to think about this more carefully.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i in {0,1}, k in {0,1,2,3}, j in {0,1,2,3}
    #
    # Split j into two groups: j in {0,1} and j in {2,3}
    # Split k into two groups: k in {0,1} and k in {2,3}
    #
    # So we have 4 sub-problems:
    # (j_group, k_group) where j_group in {0,1} and k_group in {0,1}
    # Each is 2x2x2: A[0:2, j_group*2:(j_group+1)*2] * B[j_group*2:(j_group+1)*2, k_group*2:(k_group+1)*2] -> C[0:2, k_group*2:(k_group+1)*2]
    
    blocks = []
    for jg in range(2):  # j group: 0 or 1
        for kg in range(2):  # k group: 0 or 1
            blocks.append((jg * 2, kg * 2))
    
    col = 0
    for (j_offset, k_offset) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for (li, lj), val in u_coeffs.items():
                gi = li  # row of A (0 or 1)
                gj = lj + j_offset  # col of A
                U[gi * m + gj][col] += val
            
            for (lj, lk), val in v_coeffs.items():
                gj = lj + j_offset  # row of B
                gk = lk + k_offset  # col of B
                V[gj * p + gk][col] += val
            
            for (lk, li), val in w_coeffs.items():
                # W is indexed as W[k*n + i, r]
                gi = li  # row of C (0 or 1)
                gk = lk + k_offset  # col of C
                W[gk * n + gi][col] += val
            
            col += 1
    
    # 1x4x4 part: A row 2, B 4x4, C row 2
    # C[2,k] = sum_j A[2,j] * B[j,k]
    # Rank 4: one term per output column k
    for k in range(p):
        for j in range(m):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
