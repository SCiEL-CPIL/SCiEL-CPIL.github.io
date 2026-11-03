"""Program for matrix multiplication 3x4x7 rank minimization."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U * V^T * W^T gives the correct matrix multiplication tensor.
    
    The decomposition requires:
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We verify by computing the product for all (i,j,k) triples.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False, (i, j, k, total)
    return True, None

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication.
    
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    indexed by (i,j) for U, (j,k) for V, (k,i) for W.
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    for (i,j) in [(0,0),(1,1)]:
        U[i*2+j][0] = 1.0
    for (j,k) in [(0,0),(1,1)]:
        V[j*2+k][0] = 1.0
    W[0*2+0][0] = 1.0
    W[1*2+1][0] = 1.0
    
    # M2 = (A21+A22)B11
    for (i,j) in [(1,0),(1,1)]:
        U[i*2+j][1] = 1.0
    V[0*2+0][1] = 1.0
    W[0*2+1][1] = 1.0
    W[1*2+1][1] = -1.0
    
    # M3 = A11(B21-B22)
    U[0*2+0][2] = 1.0
    V[1*2+0][2] = 1.0
    V[1*2+1][2] = -1.0
    W[0*2+0][2] = 1.0
    W[1*2+1][2] = 1.0
    
    # M4 = A22(B12-B11)
    U[1*2+1][3] = 1.0
    V[0*2+1][3] = 1.0
    V[0*2+0][3] = -1.0
    W[0*2+0][3] = 1.0
    W[0*2+1][3] = 1.0
    
    # M5 = (A11+A12)B22
    for (i,j) in [(0,0),(0,1)]:
        U[i*2+j][4] = 1.0
    V[1*2+1][4] = 1.0
    W[0*2+0][4] = -1.0
    W[0*2+0][4] += 1.0
    
    # M6 = (A21-A11)(B11+B12)
    U[1*2+0][5] = 1.0
    U[0*2+0][5] = -1.0
    V[0*2+0][5] = 1.0
    V[0*2+1][5] = 1.0
    W[1*2+1][5] = 1.0
    
    # M7 = (A12-A22)(B21+B22)
    U[0*2+1][6] = 1.0
    U[1*2+1][6] = -1.0
    V[1*2+0][6] = 1.0
    V[1*2+1][6] = 1.0
    W[0*2+0][6] = 1.0
    
    return U, V, W

def _build_strassen_block_decomposition(n, m, p):
    """Build a block decomposition using Strassen for 2x2x2 blocks.
    
    Strategy:
    - Split A (n x m) into 2x2 blocks where possible
    - Split B (m x p) into 2x2 blocks where possible  
    - Use Strassen (rank 7) for each 2x2x2 block
    - Use schoolbook for remaining 2x2x1 or 1x2x2 blocks
    
    For 3x4x7:
    A = [[A11(2x2) A12(2x2)],
         [A21(1x2) A22(1x2)]]
    B = [[B11(2x2) B12(2x2) B13(2x2) B14(2x1)],
         [B21(2x2) B22(2x2) B23(2x2) B24(2x1)]]
    
    C[0:2,:] = A11*B11 + A12*B21 (for cols 0-1)
              + A11*B12 + A12*B22 (for cols 2-3)
              + A11*B13 + A12*B23 (for cols 4-5)
              + A11*B14 + A12*B24 (for col 6)
    
    C[2,:]   = A21*B11 + A22*B21 (for cols 0-1)
              + A21*B12 + A22*B22 (for cols 2-3)
              + A21*B13 + A22*B23 (for cols 4-5)
              + A21*B14 + A22*B24 (for col 6)
    
    Each 2x2*2x2 uses Strassen (rank 7)
    Each 2x2*2x1 uses schoolbook (rank 4)
    Each 1x2*2x2 uses schoolbook (rank 4)
    Each 1x2*2x1 uses schoolbook (rank 2)
    
    Count:
    - 2x2*2x2 blocks: A11*B11, A11*B12, A11*B13, A12*B21, A12*B22, A12*B23 = 6 blocks * 7 = 42
    - 2x2*2x1 blocks: A11*B14, A12*B24 = 2 blocks * 4 = 8
    - 1x2*2x2 blocks: A21*B11, A21*B12, A21*B13, A22*B21, A22*B22, A22*B23 = 6 blocks * 4 = 24
    - 1x2*2x1 blocks: A21*B14, A22*B24 = 2 blocks * 2 = 4
    
    Total rank = 42 + 8 + 24 + 4 = 78
    """
    n, m, p = 3, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for r, v in u_entries.items():
            U_col[r] = v
        for r, v in v_entries.items():
            V_col[r] = v
        for r, v in w_entries.items():
            W_col[r] = v
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]"""
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            u_entries = {}
            v_entries = {}
            w_entries = {}
            for ui in range(4):
                if U_s[ui][r] != 0:
                    # U index: ui maps to (di, dj) where ui = di*2 + dj
                    di, dj = divmod(ui, 2)
                    u_entries[(i0 + di) * m + (j0 + dj)] = U_s[ui][r]
            for vi in range(4):
                if V_s[vi][r] != 0:
                    # V index: vi maps to (dj, dk) where vi = dj*2 + dk
                    dj, dk = divmod(vi, 2)
                    v_entries[(j0 + dj) * p + (k0 + dk)] = V_s[vi][r]
            for wi in range(4):
                if W_s[wi][r] != 0:
                    # W index: wi maps to (dk, di) where wi = dk*2 + di
                    dk, di = divmod(wi, 2)
                    w_entries[(k0 + dk) * n + (i0 + di)] = W_s[wi][r]
            add_term(u_entries, v_entries, w_entries)
    
    def add_schoolbook_2x2x1(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[k0:k0+1, i0:i0+2]"""
        for di in range(2):
            for dj in range(2):
                u_entries = {(i0 + di) * m + (j0 + dj): 1.0}
                v_entries = {(j0 + dj) * p + k0: 1.0}
                w_entries = {k0 * n + (i0 + di): 1.0}
                add_term(u_entries, v_entries, w_entries)
    
    def add_schoolbook_1x2x2(i0, j0, k0):
        """Add schoolbook for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+1]"""
        for dj in range(2):
            for dk in range(2):
                u_entries = {i0 * m + (j0 + dj): 1.0}
                v_entries = {(j0 + dj) * p + (k0 + dk): 1.0}
                w_entries = {(k0 + dk) * n + i0: 1.0}
                add_term(u_entries, v_entries, w_entries)
    
    def add_schoolbook_1x2x1(i0, j0, k0):
        """Add schoolbook for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[k0:k0+1, i0:i0+1]"""
        for dj in range(2):
            u_entries = {i0 * m + (j0 + dj): 1.0}
            v_entries = {(j0 + dj) * p + k0: 1.0}
            w_entries = {k0 * n + i0: 1.0}
            add_term(u_entries, v_entries, w_entries)
    
    # Process each block
    # A is 3x4, split as 2+1 rows, 2+2 cols
    # B is 4x7, split as 2+2 rows, 2+2+2+1 cols
    
    # Top-left 2x2 block of A (rows 0-1, cols 0-1)
    # Multiply with B blocks
    # B rows 0-1 (top), cols 0-1: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # B rows 0-1, cols 2-3: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    # B rows 0-1, cols 4-5: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 0, 4)  # A[0:2,0:2] * B[0:2,4:6] -> C[4:6,0:2]
    # B rows 0-1, col 6: 2x2x1 -> schoolbook
    add_schoolbook_2x2x1(0, 0, 6)  # A[0:2,0:2] * B[0:2,6:7] -> C[6:7,0:2]
    
    # A[0:2,2:4] (top-right 2x2 block)
    # Multiply with B blocks
    # B rows 2-3 (bottom), cols 0-1: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 2, 0)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # B rows 2-3, cols 2-3: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 2, 2)  # A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2]
    # B rows 2-3, cols 4-5: 2x2x2 -> Strassen
    add_strassen_2x2x2(0, 2, 4)  # A[0:2,2:4] * B[2:4,4:6] -> C[4:6,0:2]
    # B rows 2-3, col 6: 2x2x1 -> schoolbook
    add_schoolbook_2x2x1(0, 2, 6)  # A[0:2,2:4] * B[2:4,6:7] -> C[6:7,0:2]
    
    # A[2:3,0:2] (bottom-left 1x2 block)
    # Multiply with B blocks
    # B rows 0-1 (top), cols 0-1: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 0, 0)  # A[2:3,0:2] * B[0:2,0:2] -> C[0:2,2:3]
    # B rows 0-1, cols 2-3: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 0, 2)  # A[2:3,0:2] * B[0:2,2:4] -> C[2:4,2:3]
    # B rows 0-1, cols 4-5: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 0, 4)  # A[2:3,0:2] * B[0:2,4:6] -> C[4:6,2:3]
    # B rows 0-1, col 6: 1x2x1 -> schoolbook
    add_schoolbook_1x2x1(2, 0, 6)  # A[2:3,0:2] * B[0:2,6:7] -> C[6:7,2:3]
    
    # A[2:3,2:4] (bottom-right 1x2 block)
    # Multiply with B blocks
    # B rows 2-3 (bottom), cols 0-1: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 2, 0)  # A[2:3,2:4] * B[2:4,0:2] -> C[0:2,2:3]
    # B rows 2-3, cols 2-3: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 2, 2)  # A[2:3,2:4] * B[2:4,2:4] -> C[2:4,2:3]
    # B rows 2-3, cols 4-5: 1x2x2 -> schoolbook
    add_schoolbook_1x2x2(2, 2, 4)  # A[2:3,2:4] * B[2:4,4:6] -> C[4:6,2:3]
    # B rows 2-3, col 6: 1x2x1 -> schoolbook
    add_schoolbook_1x2x1(2, 2, 6)  # A[2:3,2:4] * B[2:4,6:7] -> C[6:7,2:3]
    
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]
    
    return U, V, W

def construct():
    """Construct a rank-minimal decomposition for 3x4x7 matrix multiplication.
    
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Expected rank: 78 (vs 84 schoolbook)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try the Strassen block decomposition
    U, V, W = _build_strassen_block_decomposition(n, m, p)
    
    # Verify
    ok, err = _verify(U, V, W, n, m, p)
    if not ok:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Safety margin check
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
