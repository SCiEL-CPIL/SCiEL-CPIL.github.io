"""Tensor decomposition for 3x3 * 3x8 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks where beneficial, combined with
schoolbook for the rest, to achieve rank < 72.
"""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix.
    
    Strategy: Partition A (3x3) into a 2x2 top-left block and remaining rows/cols.
    Use Strassen on the 2x2 block (rank 7 instead of 8), schoolbook for the rest.
    
    A is 3x3, B is 3x8, C = A*B is 3x8.
    
    Partition:
      A = [[A11, A12],
           [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
      B = [[B11, B12],
           [B21, B22]] where B11 is 2x8, B12 is 1x8, B21 is 1x8, B22 is 1x8
      
    Wait, B is 3x8, so partitioning rows of B:
      B = [[B_top (2x8)],
           [B_bottom (1x8)]]
    
    Actually let me think more carefully.
    
    A is 3x3, B is 3x8.
    C = A @ B is 3x8.
    
    Partition A into 2x2 blocks:
      A = [[A00, A01],
           [A10, A11]]
    where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1.
    
    Partition B into 2x2 blocks (splitting the 3 rows into 2+1, and 8 cols into... well, we can keep 8 as is or split):
      B = [[B00, B01],
           [B10, B11]]
    where B00 is 2x4, B01 is 2x4, B10 is 1x4, B11 is 1x4.
    
    Then:
      C00 = A00@B00 + A01@B10  (2x4)
      C01 = A00@B01 + A01@B11  (2x4)
      C10 = A10@B00 + A11@B10  (1x4)
      C11 = A10@B01 + A11@B11  (1x4)
    
    For C00 and C01, the A00@B0x part can use Strassen if we further split B0x.
    Actually, Strassen requires both matrices to be square 2x2. B00 is 2x4, not 2x2.
    
    Let me reconsider. The known result from the knowledge: "applies Strassen to all four 2x2 B-blocks for A[0:2,0:2], yielding rank 68 instead of the trivial 72."
    
    So the approach is:
    - Take A[0:2, 0:2] which is 2x2.
    - For each of the 4 columns-groups of B (since B is 3x8, split into 4 groups of 2 columns each), take B[0:2, j*2:j*2+2] which is 2x2.
    - Apply Strassen to multiply A[0:2,0:2] (2x2) by each B[0:2, j*2:j*2+2] (2x2).
    
    This gives 7 multiplications per 2x2 block instead of 8.
    There are 4 such blocks (for j=0,1,2,3), so 4*7 = 28 multiplications for the top-left 2x2 of A times the top 2 rows of B.
    
    The remaining multiplications:
    - A[0:2, 2] (2x1) times B[2, :] (1x8): 2*1*8 = 16
    - A[2, 0:2] (1x2) times B[0:2, :] (2x8): 1*2*8 = 16
    - A[2, 2] (1x1) times B[2, :] (1x8): 1*1*8 = 8
    
    Total: 28 + 16 + 16 + 8 = 68. Yes!
    
    Now let me construct the tensor decomposition.
    
    The matrix multiplication tensor T has entries T[i, j, k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k].
    
    For the schoolbook part:
    - For i in 0..2, j in 0..2, k in 0..7: term A[i,j]*B[j,k] contributes to C[i,k].
    
    For the Strassen part on A[0:2,0:2] * B[0:2, j*2:j*2+2] for each j_block in 0..3:
    
    Let me denote for block b (b=0,1,2,3), the column offset is 2*b.
    A_block = A[0:2, 0:2]
    B_block = B[0:2, 2*b:2*b+2]
    
    Strassen's 7 products:
    m1 = (A00 + A11) * (B00 + B11)
    m2 = (A10 + A11) * B00
    m3 = A00 * (B01 - B11)
    m4 = A11 * (B10 - B00)
    m5 = (A00 + A01) * B11
    m6 = (A10 - A00) * (B00 + B01)
    m7 = (A01 - A11) * (B10 + B11)
    
    Wait, standard Strassen:
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Where:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B01-B11)
    m4 = A11(B10-B00)
    m5 = (A00+A01)B11
    m6 = (A10-A00)(B00+B01)
    m7 = (A01-A11)(B10+B11)
    
    Let me verify with the knowledge note: "Fixes the Strassen W-coefficients (m5 and m7 were wrong in the original)"
    
    So I need to be careful with the W coefficients.
    
    Let me set up the tensor decomposition explicitly.
    
    For each Strassen term, I need to express it as sum of U*V*W contributions.
    
    Let me use a list of (U_index, V_index, W_index, coeff) tuples.
    
    Actually, let me think of it as: each scalar multiplication in the decomposition corresponds to one rank-1 term. The U matrix has entries in row i*m+j (for A[i,j]), V in row j*p+k (for B[j,k]), W in row k*n+i (for C[k*n+i]... wait, the convention is W[k*n+i, r]).
    
    Wait, the convention: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W has shape (p*n, rank) = (8*3, rank) = (24, rank).
    U has shape (n*m, rank) = (3*3, rank) = (9, rank).
    V has shape (m*p, rank) = (3*8, rank) = (24, rank).
    
    The decomposition satisfies: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right. The tensor for matrix multiplication is T[i,j,k] = A[i,j]*B[j,k] -> C[i,k]. So the tensor has 3 indices: i (row of A), j (col of A = row of B), k (col of B). And C[i,k] = sum_j A[i,j]*B[j,k].
    
    So T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    But the indexing in the problem is:
    U[i*m+j, r] where i is row of A, j is col of A
    V[j*p+k, r] where j is row of B, k is col of B
    W[k*n+i, r] where k is col of B, i is row of A
    
    So the decomposition is: for all i,j,k:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is the standard matrix multiplication tensor.
    
    Now for Strassen, the key insight is that we can replace 8 scalar multiplications with 7, by introducing linear combinations.
    
    For the Strassen block, instead of having separate terms for each A[i,j]*B[j,k], we group them.
    
    Let me construct the terms:
    
    For block b (0<=b<4), columns 2*b and 2*b+1 of B, rows 0 and 1 of B:
    
    The 8 schoolbook terms would be:
    A[0,0]*B[0,2b], A[0,0]*B[0,2b+1], A[0,1]*B[1,2b], A[0,1]*B[1,2b+1] -> C[0,2b], C[0,2b+1]
    A[1,0]*B[0,2b], A[1,0]*B[0,2b+1], A[1,1]*B[1,2b], A[1,1]*B[1,2b+1] -> C[1,2b], C[1,2b+1]
    
    Strassen replaces these 8 with 7 terms using linear combinations.
    
    Each Strassen "multiplication" m_s is of the form (sum of A terms) * (sum of B terms), which in the tensor decomposition becomes a rank-1 term where U has multiple non-zero entries, V has multiple non-zero entries, and W has multiple non-zero entries.
    
    Specifically, for m1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1]):
    U entries: U[0*3+0, r] = 1, U[1*3+1, r] = 1 (for the appropriate r)
    V entries: V[0*8+2b, r] = 1, V[1*8+2b+1, r] = 1
    W entries: W[(2b)*3+0, r] = 1, W[(2b+1)*3+0, r] = 1, W[(2b)*3+1, r] = 1, W[(2b+1)*3+1, r] = 1
    
    Wait, no. The W coefficient for m1 needs to contribute to C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1] with appropriate signs.
    
    C[0,2b] gets m1 + m4 - m5 + m7
    C[0,2b+1] gets m3 + m5
    C[1,2b] gets m2 + m4
    C[1,2b+1] gets m1 - m2 + m3 + m6
    
    So for m1, W contributions are:
    W[(2b)*3+0, r1] += 1  (C[0,2b])
    W[(2b+1)*3+1, r1] += 1  (C[1,2b+1])
    
    For m2, W contributions:
    W[(2b)*3+1, r2] += 1  (C[1,2b])
    W[(2b+1)*3+1, r2] -= 1  (C[1,2b+1])
    
    For m3, W contributions:
    W[(2b+1)*3+0, r3] += 1  (C[0,2b+1])
    W[(2b+1)*3+1, r3] += 1  (C[1,2b+1])
    
    For m4, W contributions:
    W[(2b)*3+0, r4] += 1  (C[0,2b])
    W[(2b)*3+1, r4] += 1  (C[1,2b])
    
    For m5, W contributions:
    W[(2b)*3+0, r5] -= 1  (C[0,2b])
    W[(2b+1)*3+0, r5] += 1  (C[0,2b+1])
    
    For m6, W contributions:
    W[(2b+1)*3+1, r6] += 1  (C[1,2b+1])
    
    For m7, W contributions:
    W[(2b)*3+0, r7] += 1  (C[0,2b])
    
    Let me double-check by expanding:
    C[0,2b] = m1 + m4 - m5 + m7
    = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    
    Where A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    B00=B[0,2b], B01=B[0,2b+1], B10=B[1,2b], B11=B[1,2b+1]
    
    Expand:
    = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    = A00*B00 + A11*B11 + A11*B10 - A11*B00 - A01*B11 + A01*B10 - A11*B10 - A11*B11
    = A00*B00 + A11*B11 - A11*B00 - A01*B11 + A01*B10 - A11*B10
    = A00*B00 + A11*B11 - A11*B00 - A01*B11 + A01*B10 - A11*B10
    
    Hmm, that should equal A00*B00 + A01*B10 (the schoolbook result for C[0,2b]).
    
    Let me recompute more carefully:
    m1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    m4 = A11(B10-B00) = A11*B10 - A11*B00
    -m5 = -(A00+A01)B11 = -A00*B11 - A01*B11
    m7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    
    Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    
    Cancel terms:
    A00*B11 - A00*B11 = 0
    A11*B00 - A11*B00 = 0
    A11*B11 - A11*B11 = 0
    A11*B10 - A11*B10 = 0
    -A01*B11 + A01*B11 = 0
    
    Remaining: A00*B00 + A01*B10 ✓
    
    Now C[0,2b+1] = m3 + m5
    m3 = A00(B01-B11) = A00*B01 - A00*B11
    m5 = (A00+A01)B11 = A00*B11 + A01*B11
    Sum: A00*B01 - A00*B11 + A00*B11 + A01*B11 = A00*B01 + A01*B11 ✓
    
    C[1,2b] = m2 + m4
    m2 = (A10+A11)B00 = A10*B00 + A11*B00
    m4 = A11(B10-B00) = A11*B10 - A11*B00
    Sum: A10*B00 + A11*B00 + A11*B10 - A11*B00 = A10*B00 + A11*B10 ✓
    
    C[1,2b+1] = m1 - m2 + m3 + m6
    m1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    -m2 = -A10*B00 - A11*B00
    m3 = A00*B01 - A00*B11
    m6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    
    Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A10*B00 + A10*B01 - A00*B00 - A00*B01
    
    Cancel:
    A00*B00 - A00*B00 = 0
    A00*B11 - A00*B11 = 0
    A11*B00 - A11*B00 = 0
    -A10*B00 + A10*B00 = 0
    A00*B01 - A00*B01 = 0
    
    Remaining: A11*B11 + A10*B01 ✓
    
    Great, so the W coefficients are:
    m1: W[(2b)*3+0] += 1, W[(2b+1)*3+1] += 1
    m2: W[(2b)*3+1] += 1, W[(2b+1)*3+1] -= 1
    m3: W[(2b+1)*3+0] += 1, W[(2b+1)*3+1] += 1
    m4: W[(2b)*3+0] += 1, W[(2b)*3+1] += 1
    m5: W[(2b)*3+0] -= 1, W[(2b+1)*3+0] += 1
    m6: W[(2b+1)*3+1] += 1
    m7: W[(2b)*3+0] += 1
    
    Now let me code this up.
    
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build U, V, W as lists of lists
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    
    # Start with empty, we'll append columns
    U_rows = []  # list of rows, each row is a list
    V_rows = []
    W_rows = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts: {row_index: coeff}"""
        for r in range(len(U_rows)):
            U_rows[r].append(u_entries.get(r, 0))
            V_rows[r].append(v_entries.get(r, 0))
            W_rows[r].append(w_entries.get(r, 0))
        # Initialize new rows if needed
        # Actually, let's initialize all rows first
    
    # Initialize
    rank_estimate = 68
    U = [[0] * rank_estimate for _ in range(n * m)]
    V = [[0] * rank_estimate for _ in range(m * p)]
    W = [[0] * rank_estimate for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A[0:2,0:2] * B[0:2, 2b:2b+2] for b in 0..3
    for b in range(4):
        col_off = 2 * b
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,col_off]+B[1,col_off+1])
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + col_off][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,col_off]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + col_off][col] = 1
        W[col_off * n + 1][col] = 1
        W[(col_off + 1) * n + 1][col] = -1
        col += 1
        
        # m3 = A[0,0] * (B[0,col_off+1]-B[1,col_off+1])
        U[0 * m + 0][col] = 1
        V[0 * p + col_off + 1][col] = 1
        V[1 * p + col_off + 1][col] = -1
        W[(col_off + 1) * n + 0][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m4 = A[1,1] * (B[1,col_off]-B[0,col_off])
        U[1 * m + 1][col] = 1
        V[1 * p + col_off][col] = 1
        V[0 * p + col_off][col] = -1
        W[col_off * n + 0][col] = 1
        W[col_off * n + 1][col] = 1
        col += 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1,col_off+1]
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = -1
        W[(col_off + 1) * n + 0][col] = 1
        col += 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,col_off]+B[0,col_off+1])
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + col_off][col] = 1
        V[0 * p + col_off + 1][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,col_off]+B[1,col_off+1])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + col_off][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = 1
        col += 1
    
    # Schoolbook for remaining terms:
    # A[0:2, 2] * B[2, :] -> 2*1*8 = 16 terms
    for i in range(2):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # A[2, 0:2] * B[0:2, :] -> 1*2*8 = 16 terms
    for j in range(2):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # A[2, 2] * B[2, :] -> 1*1*8 = 8 terms
    for k in range(p):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
    
    assert col == rank_estimate, f"Expected rank {rank_estimate}, got {col}"
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    # Verify: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # T should be all 1s
    if np.allclose(T, 1.0):
        print(f"SUCCESS: rank = {U.shape[1]}")
    else:
        print(f"FAILURE: T not all 1s. Max diff: {np.max(np.abs(T-1))}")
        print(T)
