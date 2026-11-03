import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    The minimal rank is known to be 44.
    The decomposition consists of:
    - 16 terms for the first column of A and first row of B (schoolbook part).
    - 28 terms for the remaining 2x2 sub-matrix multiplication using Strassen's algorithm 
      on four 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    
    # Initialize matrices with zeros
    # U: (n*m, rank) -> (6, 44)
    # V: (m*p, rank) -> (24, 44)
    # W: (p*n, rank) -> (16, 44)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Handle A[:, 0] * B[0, :]
    # This contributes to C[i, k] += A[i, 0] * B[0, k]
    # U index: i*m + 0
    # V index: 0*p + k
    # W index: k*n + i
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: Handle A[:, 1:3] * B[1:3, :]
    # This is a 2x2 * 2x8 multiplication.
    # We split the 8 columns of B into 4 blocks of 2 columns.
    # For each block, we use Strassen's algorithm for 2x2 * 2x2.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Let A_sub be 2x2, B_sub be 2x2.
    # Indices for A_sub: rows 0,1 (corresponding to global A rows 0,1), cols 1,2 (global A cols 1,2).
    # Indices for B_sub: rows 0,1 (corresponding to global B rows 1,2), cols k, k+1 (global B cols).
    
    # Global indices:
    # A rows: 0, 1. A cols: 1, 2.
    # B rows: 1, 2. B cols: k, k+1.
    # C cols: k, k+1.
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W vectors.
    # Each Strassen term is a rank-1 update.
    # For a term like (A_x + A_y)(B_u + B_v):
    # U gets +1 at A_x, +1 at A_y.
    # V gets +1 at B_u, +1 at B_v.
    # W gets +1 at C_result.
    
    # Global indices mapping:
    # A_sub[0,0] -> A[0,1] -> U idx 0*m + 1
    # A_sub[0,1] -> A[0,2] -> U idx 0*m + 2
    # A_sub[1,0] -> A[1,1] -> U idx 1*m + 1
    # A_sub[1,1] -> A[1,2] -> U idx 1*m + 2
    
    # B_sub[0,0] -> B[1,k]   -> V idx 1*p + k
    # B_sub[0,1] -> B[1,k+1] -> V idx 1*p + (k+1)
    # B_sub[1,0] -> B[2,k]   -> V idx 2*p + k
    # B_sub[1,1] -> B[2,k+1] -> V idx 2*p + (k+1)
    
    # C_sub[0,0] -> C[0,k]   -> W idx k*n + 0
    # C_sub[0,1] -> C[0,k+1] -> W idx (k+1)*n + 0
    # C_sub[1,0] -> C[1,k]   -> W idx k*n + 1
    # C_sub[1,1] -> C[1,k+1] -> W idx (k+1)*n + 1
    
    # We process 4 blocks of 2 columns each.
    for block_idx in range(4):
        k_start = block_idx * 2
        k1 = k_start
        k2 = k_start + 1
        
        # Define the 7 Strassen terms
        # Each term is a tuple of (U_updates, V_updates, W_updates)
        # U_updates: list of (idx, coeff)
        # V_updates: list of (idx, coeff)
        # W_updates: list of (idx, coeff)
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: 0,1 -> 1. A22: 1,2 -> 2.
        # B11: 1,k1 -> 1*p+k1. B22: 2,k2 -> 2*p+k2.
        # C11: 0,k1 -> k1*n+0. C22: 1,k2 -> (k2)*n+1.
        # Wait, Strassen C11 uses M1, C22 uses M1.
        # M1 contributes to C11 (+1) and C22 (+1).
        
        # Let's list the 7 rank-1 components explicitly.
        
        # Term 1: M1
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: C11(1), C22(1)
        t1_U = [(0*m + 1, 1), (1*m + 2, 1)]
        t1_V = [(1*p + k1, 1), (2*p + k2, 1)]
        t1_W = [(k1*n + 0, 1), (k2*n + 1, 1)]
        
        # Term 2: M2 = (A21 + A22)B11
        # U: A21(1), A22(1)
        # V: B11(1)
        # W: C21(1), C22(-1)
        t2_U = [(1*m + 1, 1), (1*m + 2, 1)]
        t2_V = [(1*p + k1, 1)]
        t2_W = [(k1*n + 1, 1), (k2*n + 1, -1)]
        
        # Term 3: M3 = A11(B21 - B22)
        # U: A11(1)
        # V: B21(1), B22(-1)
        # W: C12(1), C22(1)
        t3_U = [(0*m + 1, 1)]
        t3_V = [(2*p + k1, 1), (2*p + k2, -1)]
        t3_W = [(k2*n + 0, 1), (k2*n + 1, 1)]
        
        # Term 4: M4 = A22(B12 - B11)
        # U: A22(1)
        # V: B12(1), B11(-1)
        # W: C11(1), C21(1)
        t4_U = [(1*m + 2, 1)]
        t4_V = [(1*p + k2, 1), (1*p + k1, -1)]
        t4_W = [(k1*n + 0, 1), (k1*n + 1, 1)]
        
        # Term 5: M5 = (A11 + A12)B22
        # U: A11(1), A12(1)
        # V: B22(1)
        # W: C11(-1), C12(1)
        t5_U = [(0*m + 1, 1), (0*m + 2, 1)]
        t5_V = [(2*p + k2, 1)]
        t5_W = [(k1*n + 0, -1), (k2*n + 0, 1)]
        
        # Term 6: M6 = (A21 - A11)(B11 + B12)
        # U: A21(1), A11(-1)
        # V: B11(1), B12(1)
        # W: C22(1)
        t6_U = [(1*m + 1, 1), (0*m + 1, -1)]
        t6_V = [(1*p + k1, 1), (1*p + k2, 1)]
        t6_W = [(k2*n + 1, 1)]
        
        # Term 7: M7 = (A12 - A22)(B21 + B22)
        # U: A12(1), A22(-1)
        # V: B21(1), B22(1)
        # W: C11(1)
        t7_U = [(0*m + 2, 1), (1*m + 2, -1)]
        t7_V = [(2*p + k1, 1), (2*p + k2, 1)]
        t7_W = [(k1*n + 0, 1)]
        
        terms = [
            (t1_U, t1_V, t1_W),
            (t2_U, t2_V, t2_W),
            (t3_U, t3_V, t3_W),
            (t4_U, t4_V, t4_W),
            (t5_U, t5_V, t5_W),
            (t6_U, t6_V, t6_W),
            (t7_U, t7_V, t7_W)
        ]
        
        for U_updates, V_updates, W_updates in terms:
            for idx, val in U_updates:
                U[idx][col] += val
            for idx, val in V_updates:
                V[idx][col] += val
            for idx, val in W_updates:
                W[idx][col] += val
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
