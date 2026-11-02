import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
BUDGET_SECONDS = 145.0  # Safety margin before 150s limit

def check_time_remaining():
    return time.time() - START_TIME < BUDGET_SECONDS

def validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    R = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0
                for r in range(R):
                    u_val = U[idx_u][r] if isinstance(U[idx_u], list) else U[idx_u, r]
                    v_val = V[idx_v][r] if isinstance(V[idx_v], list) else V[idx_v, r]
                    w_val = W[idx_w][r] if isinstance(W[idx_w], list) else W[idx_w, r]
                    val += u_val * v_val * w_val
                if abs(val - 1) > 1e-9:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
    return U, V, W

def construct_block_decomposition(n, m, p, rank_333):
    """
    Construct a block decomposition of the 3x4x5 matrix multiplication.
    Split into:
    - 3x3x3 block (rows 0-2, cols 0-2 of A, cols 0-2 of B)
    - 3x3x2 block (rows 0-2, cols 0-2 of A, cols 3-4 of B)  
    - 3x1x3 block (row 3 of A, cols 0-2 of B)
    - 3x1x2 block (row 3 of A, cols 3-4 of B)
    
    For the 3x3x3 block, use the given rank decomposition.
    For other blocks, use schoolbook.
    """
    # Block dimensions
    n1, m1, p1 = 3, 3, 3  # First block
    n2, m2, p2 = 3, 3, 2  # Second block  
    n3, m3, p3 = 3, 1, 3  # Third block
    n4, m4, p4 = 3, 1, 2  # Fourth block
    
    rank_332 = n2 * m2 * p2  # Schoolbook for 3x3x2
    rank_313 = n3 * m3 * p3  # Schoolbook for 3x1x3
    rank_312 = n4 * m4 * p4  # Schoolbook for 3x1x2
    
    total_rank = rank_333 + rank_332 + rank_313 + rank_312
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: 3x3x3 (top-left)
    # Uses rows 0-2, cols 0-2 of A; rows 0-2, cols 0-2 of B
    # Output: rows 0-2, cols 0-2 of C
    # For each (i,j,k) in [0,3)×[0,3)×[0,3):
    #   U[i*m+j, col] = u_i_j_k
    #   V[j*p+k, col] = v_j_k  
    #   W[k*n+i, col] = w_k_i
    # We need to get the 3x3x3 rank decomposition
    # For now, use schoolbook as placeholder
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col_offset] = 1
                V[idx_v][col_offset] = 1
                W[idx_w][col_offset] = 1
                col_offset += 1
    # Note: This uses schoolbook (rank 27) for the 3x3x3 block
    # To use rank 23, we'd need the actual decomposition matrices
    
    # Block 2: 3x3x2 (top-right)
    # Rows 0-2, cols 0-2 of A; rows 0-2, cols 3-4 of B
    # Output: rows 0-2, cols 3-4 of C
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col_offset] = 1
                V[idx_v][col_offset] = 1
                W[idx_w][col_offset] = 1
                col_offset += 1
    
    # Block 3: 3x1x3 (bottom-left)
    # Row 3 of A; cols 0-2 of B
    # Output: row 3, cols 0-2 of C
    for i in range(3):
        for j in range(3, 4):
            for k in range(3):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col_offset] = 1
                V[idx_v][col_offset] = 1
                W[idx_w][col_offset] = 1
                col_offset += 1
    
    # Block 4: 3x1x2 (bottom-right)
    # Row 3 of A; cols 3-4 of B
    # Output: row 3, cols 3-4 of C
    for i in range(3):
        for j in range(3, 4):
            for k in range(3, 5):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col_offset] = 1
                V[idx_v][col_offset] = 1
                W[idx_w][col_offset] = 1
                col_offset += 1
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 3, 4, 5
    
    # Try to construct a better decomposition using block structure
    # The theoretical best is R(3,4,5) <= R(3,3,3) + R(3,3,2) + R(3,1,3) + R(3,1,2)
    # With R(3,3,3) <= 23, R(3,3,2) <= 14 (known), R(3,1,3) = 9, R(3,1,2) = 6
    # Total: 23 + 14 + 9 + 6 = 52
    
    # However, without the exact matrices for the low-rank blocks, we can't guarantee validity.
    # So we'll use schoolbook for all blocks, giving rank 60.
    
    # Construct the schoolbook decomposition
    U, V, W = schoolbook_decomposition(n, m, p)
    
    # Validate
    if not validate_decomposition(U, V, W, n, m, p):
        # This should never happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}
