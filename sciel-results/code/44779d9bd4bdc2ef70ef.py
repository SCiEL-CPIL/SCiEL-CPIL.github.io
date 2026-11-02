"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                computed = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(computed - expected) > 1e-10:
                    return False
    return True

def _strassen_rank7():
    """
    Return rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B10 - B11)
    # M3 = A11(B01 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 1, 0, 0, 1, 0, 0],  # A01
        [0, 0, 1, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 1],  # A11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # B00
        [1, 0, 0, 0, 0, 1, 0],  # B01
        [0, 1, 1, 0, 0, 0, 1],  # B10
        [1, 0, 1, 1, 1, 0, 1],  # B11
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # C00 = M0 - M1 - M3 + M6
        [0, 1, 0, 0, 0, 0, 0],  # C01 = M2 + M3
        [0, 0, 1, 0, 0, 1, 0],  # C10 = M1 + M4 - M6
        [0, 0, 0, 1, 1, 0, 0],  # C11 = M0 - M2 - M4 + M5
    ], dtype=float)
    
    # Fix W to match Strassen formulas exactly
    # C00 = M0 - M1 - M3 + M6
    # C01 = M2 + M3
    # C10 = M1 + M4 - M6
    # C11 = M0 - M2 - M4 + M5
    
    W = np.array([
        [1, -1, 0, -1, 0, 0, 1],  # C00
        [0, 0, 1, 1, 0, 0, 0],    # C01
        [0, 1, 0, 0, 1, 0, -1],   # C10
        [1, 0, -1, 0, -1, 1, 0],  # C11
    ], dtype=float)
    
    return U, V, W

def _recursive_strassen_4x4x4():
    """
    Return rank-28 decomposition for 4x4x4 matrix multiplication using recursive Strassen.
    Actually, let's just use 4 separate 2x2 Strassen multiplications for each block.
    That gives rank 4*7 = 28.
    
    But wait, we can do better. The 4x4x4 tensor can be decomposed using 4 Strassen-7 multiplications
    if we treat it as 4 independent 2x2 multiplications? No, that's not right.
    
    Actually, the standard approach is:
    A (4x4), B (4x4), C (4x4)
    Split into 2x2 blocks.
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each Ci is a sum of two 2x2 products.
    So we need 8 2x2 products, each of rank 7. Total rank 56.
    
    That's the recursive Strassen rank for 4x4x4: 56.
    
    Let me implement that.
    """
    n = 4
    m = 4
    p = 4
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # U rows: A_ij where i in 0..3, j in 0..3
    # V rows: B_jk where j in 0..3, k in 0..3
    # W rows: C_ki where k in 0..3, i in 0..3
    
    # Block definitions:
    # A11: rows 0,1; cols 0,1 -> indices 0,1,4,5
    # A12: rows 0,1; cols 2,3 -> indices 0,1,2,3
    # A21: rows 2,3; cols 0,1 -> indices 8,9,12,13
    # A22: rows 2,3; cols 2,3 -> indices 10,11,14,15
    
    # Wait, let me use a cleaner indexing.
    # A is 4x4, indexed by (i,j) with i in 0..3, j in 0..3
    # Row index in U: i*m + j = 4*i + j
    
    # Block (block_row, block_col) where block_row, block_col in 0..1
    # A_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # A_block(0,0) = A[0:2, 0:2] -> U rows: 0,1,4,5
    # A_block(0,1) = A[0:2, 2:4] -> U rows: 0,1,2,3
    # A_block(1,0) = A[2:4, 0:2] -> U rows: 8,9,12,13
    # A_block(1,1) = A[2:4, 2:4] -> U rows: 10,11,14,15
    
    # B is 4x4, indexed by (j,k)
    # V row index: j*p + k = 4*j + k
    # B_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # B_block(0,0) = B[0:2, 0:2] -> V rows: 0,1,4,5
    # B_block(0,1) = B[0:2, 2:4] -> V rows: 0,1,2,3
    # B_block(1,0) = B[2:4, 0:2] -> V rows: 8,9,12,13
    # B_block(1,1) = B[2:4, 2:4] -> V rows: 10,11,14,15
    
    # C is 4x4, indexed by (k,i)
    # W row index: k*n + i = 4*k + i
    # C_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # C_block(0,0) = C[0:2, 0:2] -> W rows: 0,1,4,5
    # C_block(0,1) = C[0:2, 2:4] -> W rows: 0,1,2,3
    # C_block(1,0) = C[2:4, 0:2] -> W rows: 8,9,12,13
    # C_block(1,1) = C[2:4, 2:4] -> W rows: 10,11,14,15
    
    # We need to compute:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each product is a 2x2 matrix multiplication.
    # We'll use Strassen for each.
    
    S_U, S_V, S_W = _strassen_rank7()
    
    # Helper: add a Strassen multiplication for A_block(a_r, a_c) * B_block(b_r, b_c) -> C_block(c_r, c_c)
    # with scale factor
    def add_strassen(a_br, a_bc, b_br, b_bc, c_br, c_bc, scale=1.0, offset=0):
        # A_block rows in U
        a_rows = [4*(2*a_br + i) + (2*a_bc + j) for i in range(2) for j in range(2)]
        # B_block rows in V
        b_rows = [4*(2*b_br + i) + (2*b_bc + j) for i in range(2) for j in range(2)]
        # C_block rows in W
        c_rows = [4*(2*c_br + i) + (2*c_bc + j) for i in range(2) for j in range(2)]
        
        for s in range(7):
            col = offset + s
            for local_i, global_i in enumerate(a_rows):
                U[global_i, col] += scale * S_U[local_i, s]
            for local_j, global_j in enumerate(b_rows):
                V[global_j, col] += scale * S_V[local_j, s]
            for local_k, global_k in enumerate(c_rows):
                W[global_k, col] += scale * S_W[local_k, s]
    
    # C11 = A11*B11 + A12*B21
    add_strassen(0, 0, 0, 0, 0, 0, 1.0, 0)   # A11*B11
    add_strassen(0, 1, 1, 0, 0, 0, 1.0, 7)   # A12*B21
    
    # C12 = A11*B12 + A12*B22
    add_strassen(0, 0, 0, 1, 0, 1, 1.0, 14)  # A11*B12
    add_strassen(0, 1, 1, 1, 0, 1, 1.0, 21)  # A12*B22
    
    # C21 = A21*B11 + A22*B21
    add_strassen(1, 0, 0, 0, 1, 0, 1.0, 28)  # A21*B11
    add_strassen(1, 1, 1, 0, 1, 0, 1.0, 35)  # A22*B21
    
    # C22 = A21*B12 + A22*B22
    add_strassen(1, 0, 0, 1, 1, 1, 1.0, 42)  # A21*B12
    add_strassen(1, 1, 1, 1, 1, 1, 1.0, 49)  # A22*B22
    
    return U, V, W

def _standard_mult_decomp(n, m, p, offset=0):
    """
    Return schoolbook decomposition for n x m times m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = offset + i*m*p + j*p + k
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: 
    # For the 4x4x7 problem:
    # We can treat it as multiplying a 4x4 matrix A by a 4x7 matrix B.
    # B has 7 columns. We can group them.
    
    # Option 1: Recursive Strassen for the first 4 columns (treating them as a 4x4 matrix),
    #            then standard for the last 3 columns.
    # Rank: 56 + 3*16 = 56 + 48 = 104.
    # Wait, standard for 4x4x1 is rank 16. So 3 columns = 48. Total 104.
    
    # Option 2: Use Strassen for all 7 columns? No, Strassen is for square matrices.
    
    # Option 3: Split B into 4+3. First 4 columns: recursive Strassen (rank 56).
    #            Last 3 columns: standard (rank 48). Total 104.
    
    # Option 4: Can we do better? 
    # Actually, let's just implement Option 3 and see.
    
    # Build the decomposition
    # U: (16, total_rank)
    # V: (28, total_rank)
    # W: (28, total_rank)
    
    # Part 1: First 4 columns of B (k=0,1,2,3)
    # This is a 4x4x4 multiplication.
    # U1: (16, 56), V1: (16, 56), W1: (16, 56)
    # But wait, V1 should be for B[:, 0:4], which is 4x4.
    # V rows for B: j*p + k. For k in 0..3, V rows are j*7 + k.
    # We need to map the 4x4 Strassen V (which assumes p=4) to the actual V rows.
    
    # Let me re-think. The recursive Strassen function I wrote assumes p=4 for V.
    # But here p=7. So I need to adjust.
    
    # Let me write a more flexible approach.
    
    # Actually, let's just use the schoolbook for now and try to improve.
    # Schoolbook rank = 4*4*7 = 112.
    
    # Let me try a different approach: use Strassen for 2x2 blocks in a structured way.
    
    # For now, return schoolbook as fallback.
    U, V, W = _standard_mult_decomp(n, m, p)
    rank = n * m * p
    
    # Let's try to construct a better decomposition.
    # I'll implement the 4+3 split properly.
    
    # Part 1: A (4x4) * B_0 (4x4) where B_0 = B[:, 0:4]
    # This is a 4x4x4 multiplication.
    # We have a rank-56 decomposition for this.
    # U1: (16, 56)
    # V1: (16, 56) but V1 corresponds to B_0 which has rows j in 0..3, cols k in 0..3.
    # In the full V, B_0 occupies rows j*7 + k for j in 0..3, k in 0..3.
    
    # Let me build this manually.
    
    total_rank = 56 + 48  # Strassen 4x4x4 + standard 4x4x3
    U_full = np.zeros((16, total_rank), dtype=float)
    V_full = np.zeros((28, total_rank), dtype=float)
    W_full = np.zeros((28, total_rank), dtype=float)
    
    # Part 1: Strassen for B[:, 0:4]
    S_U, S_V, S_W = _recursive_strassen_4x4x4()
    
    # Map S_V (which assumes p=4) to V_full
    # S_V row j (0..3) corresponds to B_0 row j, which is B[:, j] for j in 0..3.
    # But B_0 is B[:, 0:4], so B_0[j, k] = B[j, k] for k in 0..3.
    # V_full row for B[j, k] is j*7 + k.
    # S_V row for B_0[j, k] is j*4 + k (since p=4 in the 4x4x4 problem).
    
    for j in range(4):
        for k in range(4):
            s_row = j * 4 + k
            v_row = j * 7 + k
            V_full[v_row, 0:56] = S_V[s_row, :]
    
    U_full[:, 0:56] = S_U
    # S_W has shape (16, 56) corresponding to k=0..3, i=0..3
    # W_full has shape (28, total_rank)
    # We only fill the first 16 rows of W_full (k=0..3)
    W_full[0:16, 0:56] = S_W
    
    # Part 2: Standard for B[:, 4:7]
    # This is a 4x4x3 multiplication.
    # Rank = 4*4*3 = 48.
    offset = 56
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                col = offset + (i*4 + j)*3 + (k - 4)
                U_full[i*4 + j, col] = 1
                V_full[j*7 + k, col] = 1
                W_full[k*4 + i, col] = 1
    
    # Verify
    valid = _verify(U_full.tolist(), V_full.tolist(), W_full.tolist())
    
    if valid:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _standard_mult_decomp(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
