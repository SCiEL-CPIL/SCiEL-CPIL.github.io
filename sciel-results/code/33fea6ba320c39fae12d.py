"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact for the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check a few random matrices to ensure correctness
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_test = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # Reshape to appropriate matrix forms
            # U_r is n x m
            U_r = u_col.reshape(n, m)
            # V_r is m x p
            V_r = v_col.reshape(m, p)
            # W_r is p x n (indexed as W[k*n+i])
            W_r = w_col.reshape(p, n)
            
            # The contribution of rank-r term to C is:
            # C += (U_r @ V_r) * W_r.T  (element-wise product)
            # Note: W_r is p x n, so W_r.T is n x p.
            # (U_r @ V_r) is n x p.
            term = U_r @ V_r
            C_test += term * W_r.T
            
        if not np.allclose(C_test, C_true, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Total rank estimate:
    # C11 (2x4x4) -> 2 * R(2,2,2) = 14
    # C12 (2x4x2) -> 2 * R(2,2,2) = 14
    # C21 (1x4x4) -> R(1,4,4) = 1
    # C22 (1x4x2) -> R(1,4,2) = 1
    # Total = 30
    R = 30
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i_off: row offset in A/C (0 or 1 for the 2x2 block)
    # j_off: col offset in A / row offset in B (0 or 2 for the 4-wide dimension)
    # k_off: col offset in B/C (0, 2, 4)
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # Strassen's 7 multiplications for 2x2 matrices
        # Local indices 0,1 for the 2x2 block
        # Global indices:
        # i: i_off, i_off+1
        # j: j_off, j_off+1
        # k: k_off, k_off+1
        
        # m1 = (a+d)(e+h)
        U[i_off * m + j_off, col_start] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start] += 1
        V[j_off * p + k_off, col_start] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start] += 1
        W[k_off * n + i_off, col_start] += 1
        W[(k_off + 1) * n + (i_off + 1), col_start] += 1
        
        # m2 = (c+d)e
        U[(i_off + 1) * m + j_off, col_start + 1] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start + 1] += 1
        V[j_off * p + k_off, col_start + 1] += 1
        W[k_off * n + (i_off + 1), col_start + 1] += 1
        
        # m3 = a(f-h)
        U[i_off * m + j_off, col_start + 2] += 1
        V[j_off * p + (k_off + 1), col_start + 2] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 2] -= 1
        W[(k_off + 1) * n + i_off, col_start + 2] += 1
        
        # m4 = d(g-e)
        U[(i_off + 1) * m + (j_off + 1), col_start + 3] += 1
        V[(j_off + 1) * p + k_off, col_start + 3] += 1
        V[j_off * p + k_off, col_start + 3] -= 1
        W[k_off * n + (i_off + 1), col_start + 3] += 1
        
        # m5 = (a+b)h
        U[i_off * m + j_off, col_start + 4] += 1
        U[i_off * m + (j_off + 1), col_start + 4] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 4] += 1
        W[(k_off + 1) * n + i_off, col_start + 4] += 1
        
        # m6 = (c-a)(e+f)
        U[(i_off + 1) * m + j_off, col_start + 5] += 1
        U[i_off * m + j_off, col_start + 5] -= 1
        V[j_off * p + k_off, col_start + 5] += 1
        V[j_off * p + (k_off + 1), col_start + 5] += 1
        W[k_off * n + (i_off + 1), col_start + 5] += 1
        W[(k_off + 1) * n + (i_off + 1), col_start + 5] += 1
        
        # m7 = (b-d)(g+h)
        U[i_off * m + (j_off + 1), col_start + 6] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start + 6] -= 1
        V[(j_off + 1) * p + k_off, col_start + 6] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 6] += 1
        W[k_off * n + i_off, col_start + 6] += 1
        W[(k_off + 1) * n + i_off, col_start + 6] += 1
        
        return col_start + 7

    # Helper to add a rank-1 block for 1 x m_sub x p_sub
    # i_off: row index in A/C (must be single, e.g., 2)
    # j_off: col offset in A / row offset in B
    # k_off: col offset in B/C
    # m_sub: number of j's
    # p_sub: number of k's
    def add_rank1_block(i_off, j_off, k_off, m_sub, p_sub, col_start):
        # T[i, j, k] = 1 for j in [j_off, j_off+m_sub), k in [k_off, k_off+p_sub)
        # U[i*m + j, col] = 1
        # V[j*p + k, col] = 1
        # W[k*n + i, col] = 1
        for j in range(j_off, j_off + m_sub):
            U[i_off * m + j, col_start] = 1
        for j in range(j_off, j_off + m_sub):
            for k in range(k_off, k_off + p_sub):
                V[j * p + k, col_start] = 1
        for k in range(k_off, k_off + p_sub):
            W[k * n + i_off, col_start] = 1
        return col_start + 1

    # Block 1: C11 = A[0:2, 0:4] * B[0:4, 0:4] -> C[0:2, 0:4]
    # Split into two 2x2x2 multiplications along the middle dimension (j)
    # Sub-block 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, col)
    # Sub-block 1.2: A[0:2, 2:4] * B[2:4, 0:4] -> C[0:2, 0:2]
    # Wait, B is 4x4. The second part of B is rows 2:4, cols 0:4.
    # This is not a 2x2x2 block in the simple sense because B's cols are 0:4.
    # Let's re-evaluate the splitting of C11 (2x4x4).
    # C[0:2, 0:4] = A[0:2, 0:2]B[0:2, 0:4] + A[0:2, 2:4]B[2:4, 0:4].
    # Term 1: A_11 (2x2) * B_11 (2x4). This is 2x2x4.
    # Term 2: A_12 (2x2) * B_12 (2x4). This is 2x2x4.
    # R(2,2,4) <= 2 * R(2,2,2) = 14? No, R(2,2,4) is the rank of 2x2x4 tensor.
    # Actually, we can split B_11 (2x4) into B_11a (2x2) and B_11b (2x2).
    # So C[0:2, 0:2] gets contribution from A_11 B_11a and A_12 B_12a.
    # C[0:2, 2:4] gets contribution from A_11 B_11b and A_12 B_12b.
    
    # Let's restart the block addition with correct offsets.
    
    # Reset U, V, W and col
    U.fill(0)
    V.fill(0)
    W.fill(0)
    col = 0
    
    # --- Block C11 (2x4x4) ---
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, col)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 0, 2, col)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 2, 0, col)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, col)
    
    # --- Block C12 (2x4x2) ---
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    col = add_strassen_block(0, 0, 4, col)
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    col = add_strassen_block(0, 2, 4, col)
    
    # --- Block C21 (1x4x4) ---
    # A[2, 0:4] * B[0:4, 0:4] -> C[2, 0:4]
    col = add_rank1_block(2, 0, 0, 4, 4, col)
    
    # --- Block C22 (1x4x2) ---
    # A[2, 0:4] * B[0:4, 4:6] -> C[2, 4:6]
    col = add_rank1_block(2, 0, 4, 4, 2, col)
    
    # Verify
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, c] = 1.0
                    V_school[j * p + k, c] = 1.0
                    W_school[k * n + i, c] = 1.0
                    c += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
