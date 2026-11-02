import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _construct_block_factors(n, m, p, U, V, W, row_offset_U, row_offset_V, row_offset_W):
    """
    Constructs factors for a block of size n x m x p and adds them to the global U, V, W.
    n, m, p are the dimensions of the block.
    row_offset_U: offset for U rows (i*m + j)
    row_offset_V: offset for V rows (j*p + k)
    row_offset_W: offset for W rows (k*n + i)
    """
    pass # Placeholder, will implement specific logic below

def _strassen_2x2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (Rank 7).
    U: 4x7, V: 4x7, W: 4x7
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 34 solution.
    # Split B (3x5) into B1 (3x3) and B2 (3x2).
    # Part 1: 3x3x3 multiplication. Rank 17.
    # Part 2: 3x3x2 multiplication. Rank 17.
    
    rank_total = 34
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # --- Part 1: 3x3x3 ---
    # Embed 3x3x3 into the first 3 columns of B.
    # U indices: i*3 + j (i,j in 0..2)
    # V indices: j*3 + k (j in 0..2, k in 0..2) -> Rows 0..8
    # W indices: k*3 + i (k in 0..2, i in 0..2) -> Rows 0..8
    
    # We use the block decomposition of 3x3x3 derived earlier.
    # Blocks:
    # A22 (0,1 x 0,1), A21 (0,1 x 2), A12 (2 x 0,1), A11 (2 x 2)
    # B22 (0,1 x 0,1), B21 (0,1 x 2), B12 (2 x 0,1), B11 (2 x 2)
    
    col_offset = 0
    
    # 1. A22 B22 (2x2x2, Rank 7)
    U_22, V_22, W_22 = _strassen_2x2x2_factors()
    # Map A22 rows 0,1 -> Global 0,1. Cols 0,1 -> Global 0,1.
    # A22 indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global A indices:
    # (0,0)->0, (0,1)->1
    # (1,0)->3, (1,1)->4
    # Wait, U row index is i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    # We need to map the local 2x2 indices to global 3x3 indices.
    # Local (li, lj) -> Global (gi, gj)
    # li=0 -> gi=0, li=1 -> gi=1
    # lj=0 -> gj=0, lj=1 -> gj=1
    
    # U mapping:
    # Local U row 0 (li=0, lj=0) -> Global 0*3+0 = 0
    # Local U row 1 (li=0, lj=1) -> Global 0*3+1 = 1
    # Local U row 2 (li=1, lj=0) -> Global 1*3+0 = 3
    # Local U row 3 (li=1, lj=1) -> Global 1*3+1 = 4
    
    u_map = [0, 1, 3, 4]
    # V mapping:
    # Local V row 0 (lj=0, lk=0) -> Global 0*3+0 = 0
    # Local V row 1 (lj=0, lk=1) -> Global 0*3+1 = 1
    # Local V row 2 (lj=1, lk=0) -> Global 1*3+0 = 3
    # Local V row 3 (lj=1, lk=1) -> Global 1*3+1 = 4
    v_map = [0, 1, 3, 4]
    # W mapping:
    # Local W row 0 (lk=0, li=0) -> Global 0*3+0 = 0
    # Local W row 1 (lk=0, li=1) -> Global 0*3+1 = 1
    # Local W row 2 (lk=1, li=0) -> Global 1*3+0 = 3
    # Local W row 3 (lk=1, li=1) -> Global 1*3+1 = 4
    w_map = [0, 1, 3, 4]
    
    for c in range(7):
        for l_u, g_u in enumerate(u_map):
            U[g_u, col_offset + c] += U_22[l_u, c]
        for l_v, g_v in enumerate(v_map):
            V[g_v, col_offset + c] += V_22[l_v, c]
        for l_w, g_w in enumerate(w_map):
            W[g_w, col_offset + c] += W_22[l_w, c]
    col_offset += 7
    
    # 2. A21 B12 (2x1x2, Rank 1)
    # A21: rows 0,1; col 2.
    # B12: row 2; cols 0,1.
    # C: rows 0,1; cols 0,1.
    # C[i,k] = A[i,2] * B[2,k]
    # U: rows 0,1 (i=0,1; j=2) -> Global 0*3+2=2, 1*3+2=5
    # V: row 2 (j=2; k=0,1) -> Global 2*3+0=6, 2*3+1=7
    # W: rows 0,1 (k=0,1; i=0,1) -> Global 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    
    # Term: A[0,2]B[2,0] -> C[0,0]
    U[2, col_offset] = 1
    V[6, col_offset] = 1
    W[0, col_offset] = 1
    # Term: A[0,2]B[2,1] -> C[0,1]
    # Wait, rank 1 means one term?
    # C[i,k] = A[i,2] B[2,k].
    # This is an outer product of vector A_col2 and vector B_row2.
    # The tensor rank is 1?
    # T[i, 2, k] = 1.
    # Sum_c U[i,2,c] V[2,k,c] W[k,i,c] = 1.
    # If we set U[2,c]=1, U[5,c]=1, V[6,c]=1, V[7,c]=1, W[0,c]=1, W[1,c]=1, W[3,c]=1, W[4,c]=1?
    # No, that would make T[i,2,k] = 1 for all i,k.
    # But we need T[0,2,0]=1, T[0,2,1]=1, T[1,2,0]=1, T[1,2,1]=1.
    # And T[i,2,k] must be 0 for other i,k?
    # The tensor is defined for all i,j,k.
    # For A21 B12, only j=2, i in 0,1, k in 0,1 are non-zero.
    # So we need 4 terms?
    # Rank of 2x1x2 tensor is 4?
    # T[i,j,k] = 1 if j=2, i in 0,1, k in 0,1.
    # This is 4 distinct triples.
    # So rank is 4.
    # My previous calculation of Rank 1 was for the matrix rank, not tensor rank.
    # Tensor rank of $m \times 1 \times p$ is $m \times p$.
    # So A21 B12 has rank 4.
    
    # Let's re-calculate Rank(3x3x3).
    # 1. A22 B22: 7
    # 2. A21 B12: 2x1x2 -> Rank 4
    # 3. A22 B21: 2x2x1 -> Rank 4
    # 4. A21 B11: 2x1x1 -> Rank 2
    # 5. A12 B22: 1x2x2 -> Rank 4
    # 6. A11 B12: 1x1x2 -> Rank 2
    # 7. A12 B21: 1x2x1 -> Rank 2
    # 8. A11 B11: 1x1x1 -> Rank 1
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    
    # This is worse than 27 (schoolbook).
    # Strassen only helps for 2x2x2.
    # So the rank of 3x3x3 is likely 27 or close.
    # The "Rank 17" in the current code is likely invalid or for a different problem.
    
    # Given the difficulty of finding a valid rank < 45, and the time constraint,
    # the safest bet is to return the Schoolbook Rank 45.
    # However, the directive asks to "investigate improvements".
    # If I can't find a valid lower rank, I should stick to 45.
    
    # But wait, the current program score is 1.0.
    # If I return 45, score is 1.0.
    # If I return 44, score is 45/44 = 1.02.
    
    # Is there a simple rank 44 solution?
    # Maybe by removing one redundant term from schoolbook?
    # Unlikely.
    
    # I will implement the Schoolbook solution robustly.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
