import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target rank 28 using Strassen on 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B11 (rows 0,1 cols 0,1), B12 (rows 0,1 cols 2,3), B21 (rows 2,3 cols 0,1), B22 (rows 2,3 cols 2,3)
    # C1 (cols 0,1) = A1*B11 + A2*B21
    # C2 (cols 2,3) = A1*B12 + A2*B22
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    def add_strassen_term(col_offset, A_block_start, B_block_start, C_block_start, sign=1):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 product.
        A_block_start: (i_start, j_start) in global A
        B_block_start: (j_start, k_start) in global B
        C_block_start: (i_start, k_start) in global C
        """
        i_s, j_s = A_block_start
        j_bs, k_s = B_block_start
        i_cs, k_cs = C_block_start
        
        # Indices for 2x2 blocks
        # A: (i_s, j_s), (i_s, j_s+1), (i_s+1, j_s), (i_s+1, j_s+1)
        # B: (j_bs, k_s), (j_bs, k_s+1), (j_bs+1, k_s), (j_bs+1, k_s+1)
        # C: (i_cs, k_cs), (i_cs, k_s+1), (i_cs+1, k_cs), (i_cs+1, k_s+1)
        
        # P1: (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C12, C21, C22
        # Coeffs for U: 1, 1
        # Coeffs for V: 1, 1
        # Coeffs for W: 1, 0, 0, 1 (for C11, C12, C21, C22)
        
        # We need to be careful with indices.
        # Global indices:
        # A_idx(i, j) = i*m + j
        # B_idx(j, k) = j*p + k
        # C_idx(i, k) = k*n + i  (Trace convention: W[k*n+i, r])
        
        # P1
        col = col_offset + 0
        U[A_idx(i_s, j_s), col] += sign * 1
        U[A_idx(i_s+1, j_s+1), col] += sign * 1
        V[B_idx(j_bs, k_s), col] += sign * 1
        V[B_idx(j_bs+1, k_s+1), col] += sign * 1
        W[C_idx(i_s, k_cs), col] += sign * 1
        W[C_idx(i_s+1, k_cs+1), col] += sign * 1
        
        # P2: (A21+A22)B11
        col = col_offset + 1
        U[A_idx(i_s+1, j_s), col] += sign * 1
        U[A_idx(i_s+1, j_s+1), col] += sign * 1
        V[B_idx(j_bs, k_s), col] += sign * 1
        W[C_idx(i_s+1, k_cs), col] += sign * 1
        
        # P3: A11(B12-B22)
        col = col_offset + 2
        U[A_idx(i_s, j_s), col] += sign * 1
        V[B_idx(j_bs, k_s+1), col] += sign * 1
        V[B_idx(j_bs+1, k_s+1), col] += sign * -1
        W[C_idx(i_s, k_cs+1), col] += sign * 1
        W[C_idx(i_s+1, k_cs+1), col] += sign * 1
        
        # P4: A22(B21-B11)
        col = col_offset + 3
        U[A_idx(i_s+1, j_s+1), col] += sign * 1
        V[B_idx(j_bs+1, k_s), col] += sign * 1
        V[B_idx(j_bs, k_s), col] += sign * -1
        W[C_idx(i_s, k_cs), col] += sign * 1
        W[C_idx(i_s+1, k_cs), col] += sign * 1
        
        # P5: (A11+A12)B22
        col = col_offset + 4
        U[A_idx(i_s, j_s), col] += sign * 1
        U[A_idx(i_s, j_s+1), col] += sign * 1
        V[B_idx(j_bs+1, k_s+1), col] += sign * 1
        W[C_idx(i_s, k_cs), col] += sign * -1
        W[C_idx(i_s, k_cs+1), col] += sign * 1
        
        # P6: (A21-A11)(B11+B12)
        col = col_offset + 5
        U[A_idx(i_s+1, j_s), col] += sign * 1
        U[A_idx(i_s, j_s), col] += sign * -1
        V[B_idx(j_bs, k_s), col] += sign * 1
        V[B_idx(j_bs, k_s+1), col] += sign * 1
        W[C_idx(i_s+1, k_cs+1), col] += sign * 1
        
        # P7: (A12-A22)(B21+B22)
        col = col_offset + 6
        U[A_idx(i_s, j_s+1), col] += sign * 1
        U[A_idx(i_s+1, j_s+1), col] += sign * -1
        V[B_idx(j_bs+1, k_s), col] += sign * 1
        V[B_idx(j_bs+1, k_s+1), col] += sign * 1
        W[C_idx(i_s, k_cs), col] += sign * 1
        W[C_idx(i_s+1, k_cs), col] += sign * 1

    def A_idx(i, j):
        return i * m + j

    def B_idx(j, k):
        return j * p + k

    def C_idx(i, k):
        return k * n + i

    # Block 1: A1 (cols 0,1) * B11 (rows 0,1 cols 0,1) -> contributes to C1 (cols 0,1)
    # A1 is at i=0..1, j=0..1
    # B11 is at j=0..1, k=0..1
    # C1 is at i=0..1, k=0..1
    add_strassen_term(0, (0, 0), (0, 0), (0, 0))
    
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3 cols 0,1) -> contributes to C1 (cols 0,1)
    # A2 is at i=0..1, j=2..3
    # B21 is at j=2..3, k=0..1
    # C1 is at i=0..1, k=0..1
    add_strassen_term(7, (0, 2), (2, 0), (0, 0))
    
    # Block 3: A1 (cols 0,1) * B12 (rows 0,1 cols 2,3) -> contributes to C2 (cols 2,3)
    # A1 is at i=0..1, j=0..1
    # B12 is at j=0..1, k=2..3
    # C2 is at i=0..1, k=2..3
    add_strassen_term(14, (0, 0), (0, 2), (0, 2))
    
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3 cols 2,3) -> contributes to C2 (cols 2,3)
    # A2 is at i=0..1, j=2..3
    # B22 is at j=2..3, k=2..3
    # C2 is at i=0..1, k=2..3
    add_strassen_term(21, (0, 2), (2, 2), (0, 2))

    # Verify
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
