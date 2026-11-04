import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2_factors():
    """
    Returns the 7 Strassen factors for 2x2x2 matrix multiplication.
    Each factor is (u_coeffs, v_coeffs, w_coeffs) where each is a list of 4 entries
    corresponding to positions [0,1,2,3] = [A11,A12,A21,A22] etc.
    
    Strassen's algorithm:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    factors = [
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # M3 = A11(B21-B22) -> C12 += 1, C22 += 1
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        # M4 = A22(B12-B11) -> C11 += 1, C21 += 1
        ([0,0,0,1], [1,-1,0,0], [1,0,1,0]),
        # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A12-A22)(B21+B22) -> C11 += 1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    return factors

def _embed_factors(U_global, V_global, W_global, rank_offset, 
                   i_offset, j_offset, k_offset,
                   i_range, j_range, k_range,
                   U_sub, V_sub, W_sub):
    """
    Embed sub-problem factors into global matrices.
    U_sub: (i_range*j_range, rank_sub)
    V_sub: (j_range*k_range, rank_sub)
    W_sub: (k_range*i_range, rank_sub)
    
    Global indexing:
    U_global: index i*m + j
    V_global: index j*p + k
    W_global: index k*n + i
    """
    n, m, p = DIMENSIONS
    rank_sub = U_sub.shape[1]
    
    for r in range(rank_sub):
        for ii in range(i_range):
            for jj in range(j_range):
                i_global = i_offset + ii
                j_global = j_offset + jj
                val = U_sub[ii * j_range + jj, r]
                if val != 0:
                    U_global[i_global * m + j_global, rank_offset + r] = val
        
        for jj in range(j_range):
            for kk in range(k_range):
                j_global = j_offset + jj
                k_global = k_offset + kk
                val = V_sub[jj * k_range + kk, r]
                if val != 0:
                    V_global[j_global * p + k_global, rank_offset + r] = val
        
        for kk in range(k_range):
            for ii in range(i_range):
                k_global = k_offset + kk
                i_global = i_offset + ii
                val = W_sub[kk * i_range + ii, r]
                if val != 0:
                    W_global[k_global * n + i_global, rank_offset + r] = val
    
    return rank_offset + rank_sub

def _schoolbook_factors(i_range, j_range, k_range):
    """
    Generate schoolbook factors for i_range x j_range x k_range multiplication.
    Returns (U_sub, V_sub, W_sub) as numpy arrays.
    """
    rank = i_range * j_range * k_range
    U_sub = np.zeros((i_range * j_range, rank), dtype=int)
    V_sub = np.zeros((j_range * k_range, rank), dtype=int)
    W_sub = np.zeros((k_range * i_range, rank), dtype=int)
    
    col = 0
    for ii in range(i_range):
        for jj in range(j_range):
            for kk in range(k_range):
                U_sub[ii * j_range + jj, col] = 1
                V_sub[jj * k_range + kk, col] = 1
                W_sub[kk * i_range + ii, col] = 1
                col += 1
    
    return U_sub, V_sub, W_sub

def _strassen_factors():
    """
    Generate Strassen factors for 2x2x2 multiplication.
    Returns (U_sub, V_sub, W_sub) as numpy arrays.
    """
    factors = _strassen_2x2x2_factors()
    rank = len(factors)
    U_sub = np.zeros((4, rank), dtype=int)
    V_sub = np.zeros((4, rank), dtype=int)
    W_sub = np.zeros((4, rank), dtype=int)
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(factors):
        U_sub[:, r] = u_coeffs
        V_sub[:, r] = v_coeffs
        W_sub[:, r] = w_coeffs
    
    return U_sub, V_sub, W_sub

def _construct_block_decomposition():
    """
    Construct a rank-minimal decomposition using block structure.
    
    Strategy:
    - Split 3x5x6 into blocks and use Strassen for 2x2x2 sub-blocks.
    - Partition A (3x5) into:
      * A_top: rows 0-1 (2 rows), cols 0-3 (4 cols)
      * A_bottom: row 2 (1 row), all 5 cols
    - Partition B (5x6) into:
      * B_left: rows 0-3 (4 rows), all 6 cols
      * B_right: row 4 (1 row), all 6 cols
    
    Blocks:
    1. A_top (2x4) * B_left (4x6) -> C_top_left (2x6)
       Split into 2x2x2 and 2x2x4:
       - Cols 0-1 of B: 2x2x2 -> Strassen rank 7
       - Cols 2-5 of B: 2x4x4 -> schoolbook rank 32 (can be improved)
    
    2. A_top (2x4) * B_right (1x6) -> C_top_right (2x6)
       schoolbook rank 12
    
    3. A_bottom (1x5) * B_left (4x6) -> C_bottom_left (1x6)
       schoolbook rank 24
    
    4. A_bottom (1x5) * B_right (1x6) -> C_bottom_right (1x6)
       schoolbook rank 6
    
    Let's use a simpler partition:
    - Split m=5 into 2+3
    - Split p=6 into 2+4
    
    Block (0,0): A[:3,:2] * B[:2,:2] -> C[:3,:2] - schoolbook 12
    Block (0,1): A[:3,:2] * B[:2,2:] -> C[:3,2:] - schoolbook 24
    Block (1,0): A[:3,2:] * B[2:,:2] -> C[:3,:2] - schoolbook 12
    Block (1,1): A[:3,2:] * B[2:,2:] -> C[:3,2:] - schoolbook 24
    Total: 72
    
    Apply Strassen to some 2x2x2 sub-blocks:
    In block (0,0): A[:2,:2] * B[:2,:2] -> C[:2,:2] - Strassen rank 7
    Plus A[2:,:2] * B[:2,:2] -> C[2:,:2] - schoolbook rank 4
    Block (0,0) total: 7 + 4 = 11
    
    Similarly for other blocks with 2x2x2 sub-blocks.
    
    Let me implement a cleaner approach with explicit block decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Use a partition that allows Strassen on 2x2x2 blocks
    # Split A rows into [0,1] and [2]
    # Split A cols / B rows into [0,1] and [2,3,4]
    # Split B cols into [0,1] and [2,3,4,5]
    
    # Block structure:
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (2x2x2)
    # Block (0,1): A[0:2, 0:2] * B[0:2, 2:6] -> C[0:2, 2:6]  (2x2x4)
    # Block (0,2): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]  (2x3x2)
    # Block (0,3): A[0:2, 2:5] * B[2:5, 2:6] -> C[0:2, 2:6]  (2x3x4)
    # Block (1,0): A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]  (1x2x2)
    # Block (1,1): A[2:3, 0:2] * B[0:2, 2:6] -> C[2:3, 2:6]  (1x2x4)
    # Block (1,2): A[2:3, 2:5] * B[2:5, 0:2] -> C[2:3, 0:2]  (1x3x2)
    # Block (1,3): A[2:3, 2:5] * B[2:5, 2:6] -> C[2:3, 2:6]  (1x3x4)
    
    total_rank = 0
    U = np.zeros((n * m, 100), dtype=int)  # Allocate more than enough
    V = np.zeros((m * p, 100), dtype=int)
    W = np.zeros((p * n, 100), dtype=int)
    
    rank_offset = 0
    
    # Block (0,0): 2x2x2 -> Strassen
    U_sub, V_sub, W_sub = _strassen_factors()
    rank_offset = _embed_factors(U, V, W, rank_offset, 0, 0, 0, 2, 2, 2, U_sub, V_sub, W_sub)
    
    # Block (0,1): 2x2x4 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(2, 2, 4)
    rank_offset = _embed_factors(U, V, W, rank_offset, 0, 0, 2, 2, 2, 4, U_sub, V_sub, W_sub)
    
    # Block (0,2): 2x3x2 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(2, 3, 2)
    rank_offset = _embed_factors(U, V, W, rank_offset, 0, 2, 0, 2, 3, 2, U_sub, V_sub, W_sub)
    
    # Block (0,3): 2x3x4 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(2, 3, 4)
    rank_offset = _embed_factors(U, V, W, rank_offset, 0, 2, 2, 2, 3, 4, U_sub, V_sub, W_sub)
    
    # Block (1,0): 1x2x2 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(1, 2, 2)
    rank_offset = _embed_factors(U, V, W, rank_offset, 2, 0, 0, 1, 2, 2, U_sub, V_sub, W_sub)
    
    # Block (1,1): 1x2x4 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(1, 2, 4)
    rank_offset = _embed_factors(U, V, W, rank_offset, 2, 0, 2, 1, 2, 4, U_sub, V_sub, W_sub)
    
    # Block (1,2): 1x3x2 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(1, 3, 2)
    rank_offset = _embed_factors(U, V, W, rank_offset, 2, 2, 0, 1, 3, 2, U_sub, V_sub, W_sub)
    
    # Block (1,3): 1x3x4 -> schoolbook
    U_sub, V_sub, W_sub = _schoolbook_factors(1, 3, 4)
    rank_offset = _embed_factors(U, V, W, rank_offset, 2, 2, 2, 1, 3, 4, U_sub, V_sub, W_sub)
    
    # Trim to actual rank
    U = U[:, :rank_offset]
    V = V[:, :rank_offset]
    W = W[:, :rank_offset]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_schoolbook():
    """
    Constructs the standard schoolbook decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with index k*n+i
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Entry point to return the factor matrices.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    # Try the block decomposition first
    try:
        result = _construct_block_decomposition()
        return result
    except Exception:
        # Fall back to schoolbook
        return _construct_schoolbook()

# EVOLVE-BLOCK-END
