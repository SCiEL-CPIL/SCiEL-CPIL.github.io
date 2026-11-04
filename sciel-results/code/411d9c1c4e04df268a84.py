"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    Indices for 2x2 matrix M = [[m00, m01], [m10, m11]]:
    m00 -> 0, m01 -> 1, m10 -> 2, m11 -> 3
    
    Strassen's 7 products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)(b00)
    M3 = (a00)(b01-b11)
    M4 = (a11)(b10-b00)
    M5 = (a00+a01)(b11)
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    Result C = A*B:
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    """
    patterns = [
        # M1
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _build_strassen_decomposition(n=2, m=4, p=4):
    """
    Build a rank-28 Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
    
    The four 2x2 block products are:
    C_00 = A_00 * B_00  (A cols 0-1, B rows 0-1 cols 0-1)
    C_01 = A_00 * B_01  (A cols 0-1, B rows 0-1 cols 2-3)
    C_10 = A_01 * B_10  (A cols 2-3, B rows 2-3 cols 0-1)
    C_11 = A_01 * B_11  (A cols 2-3, B rows 2-3 cols 2-3)
    
    Note on indexing:
    A is 2x4. A_00 is A[:, 0:2], A_01 is A[:, 2:4].
    B is 4x4. B_00 is B[0:2, 0:2], B_01 is B[0:2, 2:4], B_10 is B[2:4, 0:2], B_11 is B[2:4, 2:4].
    C is 2x4. C[:, 0:2] comes from A_00*B_00 + A_01*B_10? 
    Wait, standard matrix multiplication C = A @ B.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    If we split A into left A_L (cols 0-1) and right A_R (cols 2-3):
    C = A_L @ B[:, 0:2] + A_R @ B[:, 2:3] ... No, B is split by rows corresponding to A's cols.
    
    Let's verify the block structure carefully.
    A = [A_L, A_R] where A_L is 2x2, A_R is 2x2.
    B = [B_10; B_11] where B_10 is 2x4 (top half), B_11 is 2x4 (bottom half)?
    No, B is 4x4.
    B = [[B_00, B_01], [B_10, B_11]] where each block is 2x2.
    
    C = [A_L, A_R] * [[B_00, B_01], [B_10, B_11]]
    C = [ A_L*B_00 + A_R*B_10 , A_L*B_01 + A_R*B_11 ]
    
    So C is 2x4.
    Left part of C (cols 0-1) = A_L*B_00 + A_R*B_10.
    Right part of C (cols 2-3) = A_L*B_01 + A_R*B_11.
    
    Each term like A_L*B_00 is a 2x2 matrix multiplication.
    We can compute A_L*B_00 using 7 Strassen multiplications.
    We can compute A_R*B_10 using 7 Strassen multiplications.
    Total for left part: 14 mults.
    Similarly for right part: 14 mults.
    Total rank: 28.
    
    The current code in the prompt used blocks:
    (0, 0, 0, 0),   # A_00 * B_00 -> C_00
    (0, 0, 2, 2),   # A_00 * B_01 -> C_01
    (2, 2, 0, 0),   # A_01 * B_10 -> C_10
    (2, 2, 2, 2),   # A_01 * B_11 -> C_11
    
    This maps to:
    Block 1: A_L * B_00. Contributes to C[:, 0:2].
    Block 2: A_L * B_01. Contributes to C[:, 2:4].
    Block 3: A_R * B_10. Contributes to C[:, 0:2].
    Block 4: A_R * B_11. Contributes to C[:, 2:4].
    
    This is correct.
    """
    rank = 28
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    strassen_patterns = _strassen_2x2()
    
    # Block definitions: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # a_col_offset: 0 for A_L, 2 for A_R
    # b_row_offset: 0 for top B rows, 2 for bottom B rows
    # b_col_offset: 0 for left B cols, 2 for right B cols
    # c_col_offset: 0 for left C cols, 2 for right C cols
    
    blocks = [
        (0, 0, 0, 0),   # A_L * B_00 -> C[:, 0:2]
        (0, 0, 2, 2),   # A_L * B_01 -> C[:, 2:4]
        (2, 2, 0, 0),   # A_R * B_10 -> C[:, 0:2]
        (2, 2, 2, 2),   # A_R * B_11 -> C[:, 2:4]
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for pattern in strassen_patterns:
            # Fill U: A indices
            # Local A index l in {0,1,2,3} corresponds to (r, c) = (l//2, l%2)
            # Global A index: i = r, j = a_co + c
            # U index: i*m + j
            for local_a_idx, coeff in pattern['a'].items():
                local_i = local_a_idx // 2
                local_j = local_a_idx % 2
                global_i = local_i
                global_j = a_co + local_j
                U[global_i * m + global_j][col] = coeff
            
            # Fill V: B indices
            # Local B index l in {0,1,2,3} corresponds to (r, c) = (l//2, l%2)
            # Global B index: j = b_ro + r, k = b_co + c
            # V index: j*p + k
            for local_b_idx, coeff in pattern['b'].items():
                local_j = local_b_idx // 2
                local_k = local_b_idx % 2
                global_j = b_ro + local_j
                global_k = b_co + local_k
                V[global_j * p + global_k][col] = coeff
            
            # Fill W: C indices (trace convention W[k*n+i])
            # Local C index l in {0,1,2,3} corresponds to (r, c) = (l//2, l%2)
            # Global C index: i = r, k = c_co + c
            # W index: k*n + i
            for local_c_idx, coeff in pattern['c'].items():
                local_i = local_c_idx // 2
                local_k = local_c_idx % 2
                global_i = local_i
                global_k = c_co + local_k
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return U, V, W


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    
    Interface:
      U shape: (n*m, rank) = (8, rank)
      V shape: (m*p, rank) = (16, rank)  
      W shape: (p*n, rank) = (8, rank)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention)
    """
    t0 = time.time()
    deadline = 115.0  # seconds, leave 5s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Ensure we're within the time limit
    if time.time() - t0 > deadline:
        # This should not happen as the construction is very fast
        pass
    
    return {"U": U, "V": V, "W": W}
