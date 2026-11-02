import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank: 36 (Same as parent, but distinct implementation)
    # Strategy: Vectorized construction using NumPy for clarity and speed.
    # Decomposition:
    # 1. Split B into B_left (4x4) and B_right (4x1).
    # 2. Compute C_left = A * B_left using 4 Strassen 2x2x2 blocks (Rank 28).
    # 3. Compute C_right = A * B_right using 8 Schoolbook terms (Rank 8).
    # Total Rank = 36.
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add a Strassen 2x2x2 block at specified offsets."""
        nonlocal col
        # U indices: (row, col) of A -> idx = row*m + col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        # V indices: (row, col) of B -> idx = row*p + col
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        # W indices: (row, col) of C -> idx = col*n + row (Trace convention W[k*n+i,r])
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00+a11)(b00+b11) -> C00, C11
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2 = (a10+a11)b00 -> C10, C11
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3 = a00(b01-b11) -> C01, C11
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4 = a11(b10-b00) -> C00, C10
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5 = (a00+a01)b11 -> C00, C01
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01) -> C11
        U[u10, col] += 1; U[u00, col] -= 1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11) -> C00
        U[u01, col] += 1; U[u11, col] -= 1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_2x4_1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add Schoolbook terms for 2x4 * 4x1 -> 2x1.
        A is 2x4, B is 4x1, C is 2x1.
        C0 = sum(A0k * Bk)
        C1 = sum(A1k * Bk)
        Total 8 terms.
        """
        nonlocal col
        # W indices for C (2x1)
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        for k in range(4):
            # Term for C0: A[0, k] * B[k, 0]
            u_idx = a_r0 * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w0, col] = 1
            col += 1
            
            # Term for C1: A[1, k] * B[k, 0]
            u_idx = (a_r0 + 1) * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w1, col] = 1
            col += 1

    # --- Construction ---
    
    # Part 1: C[:, 0:4] = A * B[:, 0:4]
    # Split A into A_top (1x4) and A_bot (1x4)? No, Strassen needs 2x2.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B[:, 0:4] into B_TL (2x2), B_TR (2x2) for top rows?
    # B is 4x4.
    # B_TL: rows 0-1, cols 0-1.
    # B_TR: rows 0-1, cols 2-3.
    # B_BL: rows 2-3, cols 0-1.
    # B_BR: rows 2-3, cols 2-3.
    
    # C = A_L B_TL + A_L B_TR + A_R B_BL + A_R B_BR
    # A_L is cols 0-1 of A. A_R is cols 2-3 of A.
    
    add_strassen_block(0, 0, 0, 0, 0, 0) # A_L * B_TL
    add_strassen_block(0, 0, 0, 2, 0, 2) # A_L * B_TR
    add_strassen_block(0, 2, 2, 0, 0, 0) # A_R * B_BL
    add_strassen_block(0, 2, 2, 2, 0, 2) # A_R * B_BR
    
    # Part 2: C[:, 4] = A * B[:, 4]
    # B[:, 4] is 4x1.
    add_schoolbook_2x4_1(0, 0, 0, 4, 0, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
