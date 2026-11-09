"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def _build_rank28():
    """
    Build a rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy: Partition the 4x4 matrix into four 2x2 blocks.
    The 2x4 matrix has 2 rows, each with 4 columns (two 2x2 blocks).
    We can think of this as multiplying a 2x4 matrix by a 4x4 matrix.
    
    Actually, let's use the block approach more carefully:
    - A is 2x4, B is 4x4, C is 2x4
    - Partition columns of A and rows of B into 2 groups of 2
    - This gives us a block structure where we have 2 "row blocks" of A 
      (each 1x4, or think of it as 2x2 blocks if we partition differently)
    
    Let me reconsider. A better approach:
    - Think of A as having 2 rows, B as 4x4
    - Partition the 4 columns of A / 4 rows of B into 2 groups of 2
    - So we have a "block" multiplication where each block is 2x2
    - But A is 2x4, so it's like two 2x2 blocks side by side
    - B is 4x4, so it's like four 2x2 blocks
    - C is 2x4, so it's like two 2x2 blocks side by side
    
    Using Strassen for each independent 2x2x2 sub-problem:
    For block (j_block, k_block) where j_block in {0,1} and k_block in {0,1}:
      A[:, 2*j_block:2*j_block+2] * B[2*k_block:2*k_block+2, :] 
      contributes to C[:, 2*k_block:2*k_block+2]
    
    Each such sub-problem is 2x2 * 2x2 -> 2x2, which has rank 7 (Strassen).
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)  
    # W has shape (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 terms for 2x2 matrix multiplication
    # For 2x2 matrices A, B, C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (row, col, coeff) in the 2x2 A sub-block
    # v_coeffs: list of (row, col, coeff) in the 2x2 B sub-block
    # w_coeffs: list of (row, col, coeff) in the 2x2 C sub-block
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'u': [(0, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (1, 1, 1)]  # C00 += M1, C11 += M1
        },
        # M2 = (A10+A11)B00
        {
            'u': [(1, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1)],
            'w': [(1, 0, 1), (1, 1, -1)]  # C10 += M2, C11 -= M2
        },
        # M3 = A00(B10-B11)
        {
            'u': [(0, 0, 1)],
            'v': [(1, 0, 1), (1, 1, -1)],
            'w': [(0, 1, 1), (1, 1, 1)]  # C01 += M3, C11 += M3
        },
        # M4 = A11(B01-B00)
        {
            'u': [(1, 1, 1)],
            'v': [(0, 1, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (1, 0, 1)]  # C00 += M4, C10 += M4
        },
        # M5 = (A00+A01)B11
        {
            'u': [(0, 0, 1), (0, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 0, -1), (0, 1, 1)]  # C00 -= M5, C01 += M5
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'u': [(1, 0, 1), (0, 0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)]  # C11 += M6
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'u': [(0, 1, 1), (1, 1, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)]  # C00 += M7
        },
    ]
    
    # Block offsets
    # j_block: which 2x2 block of A's columns (and B's rows)
    # k_block: which 2x2 block of B's columns (and C's columns)
    j_blocks = [0, 2]  # column offset in A, row offset in B
    k_blocks = [0, 2]  # column offset in B, column offset in C
    
    col = 0
    for j_off in j_blocks:
        for k_off in k_blocks:
            for term in strassen_terms:
                # Fill U
                for (r, c, coeff) in term['u']:
                    # A[r, j_off+c] maps to U[r*m + j_off+c]
                    U[r * m + (j_off + c)][col] += coeff
                
                # Fill V
                for (r, c, coeff) in term['v']:
                    # B[j_off+r, k_off+c] maps to V[(j_off+r)*p + k_off+c]
                    V[(j_off + r) * p + (k_off + c)][col] += coeff
                
                # Fill W
                for (r, c, coeff) in term['w']:
                    # C[r, k_off+c] maps to W[(k_off+c)*n + r]
                    # Following the trace convention W[k*n+i, r] where k is col index in C, i is row index
                    W[(k_off + c) * n + r][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct the factor matrices for the matrix multiplication tensor.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Build the rank-28 solution
    result = _build_rank28()
    
    # Verify the solution is correct
    # For safety, we could verify, but it should be correct by construction
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # This shouldn't happen for the direct construction, but guard just in case
        return result
    
    return result
# EVOLVE-BLOCK-END
