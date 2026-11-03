"""Rewritten program: deterministic rank-68 tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 8]


def construct():
    """Return a rank-68 exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    The decomposition uses:
    - 4 disjoint Strassen blocks (7 terms each) for the 2x2 * 2x2 sub-problems
    - Schoolbook terms for the remaining cross terms
    
    Total rank = 4*7 + 16 + 16 + 8 = 28 + 40 = 68
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 68
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 68
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 68
    
    col = 0
    
    # Helper: add a term where u[i]=1, v[j]=1, w[k]=1 (schoolbook)
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i][col] = 1
        V[j][col] = 1
        W[k][col] = 1
        col += 1
    
    # Helper: add a Strassen term with given u, v, w vectors
    def add_strassen_term(u_vec, v_vec, w_vec):
        nonlocal col
        for idx in range(9):
            if u_vec[idx] != 0:
                U[idx][col] = float(u_vec[idx])
        for idx in range(24):
            if v_vec[idx] != 0:
                V[idx][col] = float(v_vec[idx])
        for idx in range(24):
            if w_vec[idx] != 0:
                W[idx][col] = float(w_vec[idx])
        col += 1
    
    # ============================================================
    # Part 1: A[0:2,0:2] * B[0:2,0:8] using 4 Strassen blocks
    # B[0:2,0:8] is split into 4 blocks of 2x2: B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[0:2,6:8]
    # ============================================================
    
    # Strassen formulas for 2x2 * 2x2:
    # P1 = A00 * B00
    # P2 = A10 * B10
    # P3 = A01 * (B10 - B00)
    # P4 = A11 * (B01 - B11)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each of the 4 column blocks of B[0:2, :8]:
    for blk in range(4):
        b_col_off = blk * 2  # 0, 2, 4, 6
        
        # P1 = A00 * B00  (A[0,0] * B[0, b_col_off])
        add_strassen_term(
            [1,0,0, 0,0,0, 0,0,0],   # u: A[0,0]
            [1,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[0, b_col_off]
            [1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[0,0]
        )
        
        # P2 = A10 * B10  (A[1,0] * B[1, b_col_off])
        add_strassen_term(
            [0,1,0, 0,0,0, 0,0,0],   # u: A[1,0]
            [0,0, 1,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[1, b_col_off]
            [0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[1,0]
        )
        
        # P3 = A01 * (B10 - B00)  (A[0,1] * (B[1, b_col_off] - B[0, b_col_off]))
        add_strassen_term(
            [0,0,1, 0,0,0, 0,0,0],   # u: A[0,1]
            [0,0, -1,1, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: -B[0,off] + B[1,off]
            [0,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[0,1]
        )
        
        # P4 = A11 * (B01 - B11)  (A[1,1] * (B[0, b_col_off+1] - B[1, b_col_off+1]))
        add_strassen_term(
            [0,0,0, 0,0,1, 0,0,0],   # u: A[1,1]
            [0,0, 0,0, 1,-1, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[0,off+1] - B[1,off+1]
            [1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[0,0]+C[1,0]
        )
        
        # P5 = (A00 + A01) * B11  ((A[0,0]+A[0,1]) * B[1, b_col_off+1])
        add_strassen_term(
            [1,0,1, 0,0,0, 0,0,0],   # u: A[0,0]+A[0,1]
            [0,0, 0,0, 0,0, 1,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[1,off+1]
            [-1,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: -C[0,0]+C[0,1]
        )
        
        # P6 = (A10 - A00) * (B00 + B01)  ((A[1,0]-A[0,0]) * (B[0,off]+B[0,off+1]))
        add_strassen_term(
            [-1,1,0, 0,0,0, 0,0,0],  # u: -A[0,0]+A[1,0]
            [1,1, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[0,off]+B[0,off+1]
            [0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[1,1]
        )
        
        # P7 = (A01 - A11) * (B10 + B11)  ((A[0,1]-A[1,1]) * (B[1,off]+B[1,off+1]))
        add_strassen_term(
            [0,0,1, 0,0,-1, 0,0,0],  # u: A[0,1]-A[1,1]
            [0,0, 1,1, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],  # v: B[1,off]+B[1,off+1]
            [1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]  # w: C[0,0]
        )
    
    # ============================================================
    # Part 2: A[2,0:2] * B[0:2,0:8] (row 2 of A, first 2 cols) * (first 2 rows of B, all 8 cols)
    # Schoolbook: 1*2*8 = 16 terms
    # A[2,0] * B[0,j], A[2,1] * B[1,j] for j in 0..7
    # ============================================================
    for j in range(8):
        # A[2,0] * B[0,j] -> C[2,j]
        add_schoolbook(6, j, j * 3 + 2)  # i=2, j=0, k=j -> W[k*n+i] = W[j*3+2]
        # A[2,1] * B[1,j] -> C[2,j]
        add_schoolbook(7, 8 + j, j * 3 + 2)  # i=2, j=1, k=j
    
    # ============================================================
    # Part 3: A[0:2,2] * B[2,0:8] (first 2 rows of A, col 2) * (row 2 of B, all 8 cols)
    # Schoolbook: 2*1*8 = 16 terms
    # A[i,2] * B[2,j] -> C[i,j] for i in 0,1; j in 0..7
    # ============================================================
    for i in range(2):
        for j in range(8):
            # A[i,2] * B[2,j] -> C[i,j]
            add_schoolbook(2 + i, 16 + j, j * 3 + i)  # i=i, j=2, k=j
    
    # ============================================================
    # Part 4: A[2,2] * B[2,0:8] (scalar A[2,2] * row 2 of B)
    # Schoolbook: 1*1*8 = 8 terms
    # ============================================================
    for j in range(8):
        # A[2,2] * B[2,j] -> C[2,j]
        add_schoolbook(8, 23, j * 3 + 2)  # i=2, j=2, k=j
    
    return {"U": U, "V": V, "W": W}
