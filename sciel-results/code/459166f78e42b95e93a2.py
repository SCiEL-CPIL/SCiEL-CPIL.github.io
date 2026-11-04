"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook_block(n, m, p, i_start, i_end, j_start, j_end, k_start, k_end, offset=0):
    """Build schoolbook terms for a sub-block of indices."""
    U_rows = []
    V_rows = []
    W_rows = []
    col = 0
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                U_rows.append((i * m + j, 1))
                V_rows.append((j * p + k, 1))
                W_rows.append((k * n + i, 1))
                col += 1
    return U_rows, V_rows, W_rows, col

def _build_strassen_block(n, m, p, i_start, i_end, j_start, j_end, k_start, k_end, offset=0):
    """
    Build Strassen terms for a 2x2x2 sub-block starting at (i_start, j_start, k_start).
    The sub-block covers i in [i_start, i_start+2), j in [j_start, j_start+2), k in [k_start, k_start+2).
    
    Strassen for 2x2x2 has rank 7.
    The tensor is T[i,j,k] = delta_{i,j,k} for i,j,k in {0,1} relative to start.
    
    Standard Strassen decomposition for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Where A is 2x2 with A[i,j], B is 2x2 with B[j,k], result C[i,k].
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    """
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Indices in the sub-block
    # i indices: i_start, i_start+1
    # j indices: j_start, j_start+1
    # k indices: k_start, k_start+1
    
    # M1 = (A00+A11)(B00+B11)
    # U: rows (i0,j0) and (i1,j1) get +1
    # V: rows (j0,k0) and (j1,k1) get +1
    # W: rows (k0,i0)+1, (k0,i1)-1, (k1,i0)-1, (k1,i1)+1
    # Wait, let me re-derive. C[i,k] = sum_j A[i,j]*B[j,k]
    # M1 contributes to C00: +1, C01: 0, C10: 0, C11: +1? No.
    
    # Let me be more careful. The Strassen algorithm computes:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each M, the U factor has entries for A, V for B, W for C.
    
    # M1: A part: A00+A11 -> U[(i0,j0)]=1, U[(i1,j1)]=1
    #     B part: B00+B11 -> V[(j0,k0)]=1, V[(j1,k1)]=1
    #     C part: contributes to C00 (+1) and C11 (+1)
    #     So W[(k0,i0)]=1, W[(k1,i1)]=1
    U_rows.append((i_start * m + j_start, 1))
    U_rows.append(((i_start+1) * m + (j_start+1), 1))
    V_rows.append((j_start * p + k_start, 1))
    V_rows.append(((j_start+1) * p + (k_start+1), 1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append(((k_start+1) * n + (i_start+1), 1))
    
    # M2: A part: A10+A11 -> U[(i1,j0)]=1, U[(i1,j1)]=1
    #     B part: B00 -> V[(j0,k0)]=1
    #     C part: contributes to C10 (+1) and C11 (-1)
    #     W[(k0,i1)]=1, W[(k1,i1)]=-1
    U_rows.append(((i_start+1) * m + j_start, 1))
    U_rows.append(((i_start+1) * m + (j_start+1), 1))
    V_rows.append((j_start * p + k_start, 1))
    W_rows.append((k_start * n + (i_start+1), 1))
    W_rows.append(((k_start+1) * n + (i_start+1), -1))
    
    # M3: A part: A00 -> U[(i0,j0)]=1
    #     B part: B10-B11 -> V[(j1,k0)]=1, V[(j1,k1)]=-1
    #     C part: contributes to C01 (+1) and C11 (+1)
    #     W[(k1,i0)]=1, W[(k1,i1)]=1
    U_rows.append((i_start * m + j_start, 1))
    V_rows.append(((j_start+1) * p + k_start, 1))
    V_rows.append(((j_start+1) * p + (k_start+1), -1))
    W_rows.append(((k_start+1) * n + i_start, 1))
    W_rows.append(((k_start+1) * n + (i_start+1), 1))
    
    # M4: A part: A11 -> U[(i1,j1)]=1
    #     B part: B01-B10 -> V[(j0,k1)]=1, V[(j1,k0)]=-1
    #     C part: contributes to C00 (+1) and C10 (+1)
    #     W[(k0,i0)]=1, W[(k0,i1)]=1
    U_rows.append(((i_start+1) * m + (j_start+1), 1))
    V_rows.append((j_start * p + (k_start+1), 1))
    V_rows.append(((j_start+1) * p + k_start, -1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append((k_start * n + (i_start+1), 1))
    
    # M5: A part: A00+A01 -> U[(i0,j0)]=1, U[(i0,j1)]=1
    #     B part: B11 -> V[(j1,k1)]=1
    #     C part: contributes to C01 (+1) and C00 (-1)
    #     W[(k1,i0)]=1, W[(k0,i0)]=-1
    U_rows.append((i_start * m + j_start, 1))
    U_rows.append((i_start * m + (j_start+1), 1))
    V_rows.append(((j_start+1) * p + (k_start+1), 1))
    W_rows.append(((k_start+1) * n + i_start, 1))
    W_rows.append((k_start * n + i_start, -1))
    
    # M6: A part: A10-A00 -> U[(i1,j0)]=1, U[(i0,j0)]=-1
    #     B part: B00+B01 -> V[(j0,k0)]=1, V[(j0,k1)]=1
    #     C part: M6 is not used in the standard Strassen formulas above!
    #     Wait, I listed 7 M's but the standard Strassen uses 7. Let me check.
    #     Actually, M6 is typically defined but let me verify the formulas.
    #     
    #     Standard Strassen:
    #     M1 = (A00+A11)(B00+B11)
    #     M2 = (A10+A11)B00
    #     M3 = A00(B10-B11)
    #     M4 = A11(B01-B10)
    #     M5 = (A00+A01)B11
    #     M6 = (A10-A00)(B00+B01)
    #     M7 = (A01-A11)(B10+B11)
    #     
    #     C00 = M1 + M4 - M5 - M7
    #     C01 = M3 + M5
    #     C10 = M2 + M4
    #     C11 = M1 - M2 + M3 + M7
    #     
    #     Hmm, M6 doesn't appear in the C formulas. That's strange. Let me re-check.
    #     Actually, I think I may have the formulas slightly off. Let me use a different reference.
    #     
    #     Alternative standard Strassen:
    #     M1 = (A00+A11)(B00+B11)
    #     M2 = (A10+A11)B00
    #     M3 = A00(B10-B11)
    #     M4 = A11(B01-B10)
    #     M5 = (A00+A01)B11
    #     M6 = (A10-A00)(B00+B01)
    #     M7 = (A01-A11)(B10+B11)
    #     
    #     C00 = M1 + M4 - M5 - M7
    #     C01 = M3 + M5
    #     C10 = M2 + M4
    #     C11 = M1 - M2 + M3 + M7
    #     
    #     This doesn't use M6. That can't be right for a rank-7 decomposition.
    #     
    #     Let me look up the correct Strassen formulas. The standard ones are:
    #     P1 = (A11 - A22)(B21 + B22)
    #     P2 = (A11 + A22)(B11 + B22)
    #     P3 = (A12 - A22)B21
    #     P4 = (A11 - A12)B12
    #     P5 = A11(B11 - B21)
    #     P6 = A22(B22 - B11)
    #     P7 = (A11 + A12)(B12 - B22)
    #     
    #     C11 = P5 + P6 + P7 - P2 + P4
    #     C12 = P1 + P2 - P6 + P3
    #     C21 = P1 + P5 - P7 + P2
    #     C22 = P3 + P4 - P1 + P6
    #     
    #     Let me use this version. I'll map my indices:
    #     A11 -> A[0,0] (i0,j0), A12 -> A[0,1] (i0,j1)
    #     A21 -> A[1,0] (i1,j0), A22 -> A[1,1] (i1,j1)
    #     B11 -> B[0,0] (j0,k0), B12 -> B[0,1] (j0,k1)
    #     B21 -> B[1,0] (j1,k0), B22 -> B[1,1] (j1,k1)
    #     C11 -> C[0,0] (i0,k0), C12 -> C[0,1] (i0,k1)
    #     C21 -> C[1,0] (i1,k0), C22 -> C[1,1] (i1,k1)
    
    # Let me restart with the correct P1-P7 formulas.
    # I'll clear the previous M1-M5 entries and redo.
    
    # Actually, let me just build from scratch with the P1-P7 scheme.
    U_rows = []
    V_rows = []
    W_rows = []
    
    # P1 = (A21 - A22)(B11 + B12)  [using 1-indexed: A21=A[1,0], A22=A[1,1], B11=B[0,0], B12=B[0,1]]
    # U: (i1,j0)=1, (i1,j1)=-1
    # V: (j0,k0)=1, (j0,k1)=1
    # C contributions: C11: -P1, C12: +P1, C21: +P1, C22: -P1
    # W: (k0,i0)=-1, (k1,i0)=+1, (k0,i1)=+1, (k1,i1)=-1
    U_rows.append(((i_start+1) * m + j_start, 1))
    U_rows.append(((i_start+1) * m + (j_start+1), -1))
    V_rows.append((j_start * p + k_start, 1))
    V_rows.append((j_start * p + (k_start+1), 1))
    W_rows.append((k_start * n + i_start, -1))
    W_rows.append(((k_start+1) * n + i_start, 1))
    W_rows.append((k_start * n + (i_start+1), 1))
    W_rows.append(((k_start+1) * n + (i_start+1), -1))
    
    # P2 = (A11 + A22)(B11 + B22)
    # U: (i0,j0)=1, (i1,j1)=1
    # V: (j0,k0)=1, (j1,k1)=1
    # C contributions: C11: -P2, C12: +P2, C21: +P2, C22: 0
    # W: (k0,i0)=-1, (k1,i0)=+1, (k0,i1)=+1, (k1,i1)=0
    U_rows.append((i_start * m + j_start, 1))
    U_rows.append(((i_start+1) * m + (j_start+1), 1))
    V_rows.append((j_start * p + k_start, 1))
    V_rows.append(((j_start+1) * p + (k_start+1), 1))
    W_rows.append((k_start * n + i_start, -1))
    W_rows.append(((k_start+1) * n + i_start, 1))
    W_rows.append((k_start * n + (i_start+1), 1))
    
    # P3 = (A12 - A22)B21
    # U: (i0,j1)=1, (i1,j1)=-1
    # V: (j1,k0)=1
    # C contributions: C11: 0, C12: +P3, C21: 0, C22: +P3
    # W: (k0,i0)=0, (k1,i0)=+1, (k0,i1)=0, (k1,i1)=+1
    U_rows.append((i_start * m + (j_start+1), 1))
    U_rows.append(((i_start+1) * m + (j_start+1), -1))
    V_rows.append(((j_start+1) * p + k_start, 1))
    W_rows.append(((k_start+1) * n + i_start, 1))
    W_rows.append(((k_start+1) * n + (i_start+1), 1))
    
    # P4 = (A11 - A12)B12
    # U: (i0,j0)=1, (i0,j1)=-1
    # V: (j0,k1)=1
    # C contributions: C11: +P4, C12: 0, C21: 0, C22: +P4
    # W: (k0,i0)=+1, (k1,i0)=0, (k0,i1)=0, (k1,i1)=+1
    U_rows.append((i_start * m + j_start, 1))
    U_rows.append((i_start * m + (j_start+1), -1))
    V_rows.append((j_start * p + (k_start+1), 1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append(((k_start+1) * n + (i_start+1), 1))
    
    # P5 = A11(B11 - B21)
    # U: (i0,j0)=1
    # V: (j0,k0)=1, (j1,k0)=-1
    # C contributions: C11: +P5, C12: 0, C21: +P5, C22: 0
    # W: (k0,i0)=+1, (k1,i0)=0, (k0,i1)=+1, (k1,i1)=0
    U_rows.append((i_start * m + j_start, 1))
    V_rows.append((j_start * p + k_start, 1))
    V_rows.append(((j_start+1) * p + k_start, -1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append((k_start * n + (i_start+1), 1))
    
    # P6 = A22(B22 - B11)
    # U: (i1,j1)=1
    # V: (j1,k1)=1, (j0,k0)=-1
    # C contributions: C11: +P6, C12: 0, C21: 0, C22: +P6
    # W: (k0,i0)=+1, (k1,i0)=0, (k0,i1)=0, (k1,i1)=+1
    U_rows.append(((i_start+1) * m + (j_start+1), 1))
    V_rows.append(((j_start+1) * p + (k_start+1), 1))
    V_rows.append((j_start * p + k_start, -1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append(((k_start+1) * n + (i_start+1), 1))
    
    # P7 = (A11 + A12)(B12 - B22)
    # U: (i0,j0)=1, (i0,j1)=1
    # V: (j0,k1)=1, (j1,k1)=-1
    # C contributions: C11: +P7, C12: 0, C21: -P7, C22: 0
    # W: (k0,i0)=+1, (k1,i0)=0, (k0,i1)=-1, (k1,i1)=0
    U_rows.append((i_start * m + j_start, 1))
    U_rows.append((i_start * m + (j_start+1), 1))
    V_rows.append((j_start * p + (k_start+1), 1))
    V_rows.append(((j_start+1) * p + (k_start+1), -1))
    W_rows.append((k_start * n + i_start, 1))
    W_rows.append((k_start * n + (i_start+1), -1))
    
    return U_rows, V_rows, W_rows, 7

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Strategy: Use Strassen for the 2x4x8 part (rows 1-2 of A) and schoolbook for row 0.
    # Row 0 of A: schoolbook for 1x4x8 -> rank 32
    # Rows 1-2 of A: Strassen-based for 2x4x8
    #   For 2x4x8, we can tile: 2x4x8 = 2x2x2 blocks
    #   j has 4 columns, k has 8 columns
    #   Tile into 2x2x2 blocks: j: 2 blocks, k: 4 blocks -> 8 blocks of 2x2x2
    #   Each 2x2x2 uses Strassen rank 7 -> 8*7 = 56
    #   Total: 32 + 56 = 88
    
    U_entries = {}
    V_entries = {}
    W_entries = {}
    
    def add_term(u_row, v_row, w_row, coeff):
        key_u = (u_row,)
        key_v = (v_row,)
        key_w = (w_row,)
        # We need to handle multiple terms. Let's use a list of (row, coeff) per column.
        # Actually, let's just collect all (row, coeff) and later merge.
        pass
    
    # Better approach: collect all (row, coeff) tuples for each of U, V, W, grouped by column.
    # Each "term" (column) has one U row, one V row, one W row with some coefficients.
    # But Strassen has multiple non-zero entries per factor per term.
    
    # Let me use a different approach: for each rank term, collect the non-zero entries.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # I'll build lists of (row, col, val) for each matrix, then convert to dense.
    
    U_spare = []  # (row, col, val)
    V_spare = []
    W_spare = []
    
    col = 0
    
    # Part 1: Schoolbook for i=0 (row 0 of A), all j in 0..3, all k in 0..7
    for j in range(m):
        for k in range(p):
            U_spare.append((0 * m + j, col, 1))
            V_spare.append((j * p + k, col, 1))
            W_spare.append((k * n + 0, col, 1))
            col += 1
    
    # Part 2: Strassen for i=1,2 (rows 1-2 of A), all j in 0..3, all k in 0..7
    # Tile j into 2 blocks of 2, k into 4 blocks of 2
    for jb in range(m // 2):  # 2 blocks
        for kb in range(p // 2):  # 4 blocks
            j_start = jb * 2
            k_start = kb * 2
            i_start = 1  # rows 1 and 2
            
            u_rows, v_rows, w_rows, r = _build_strassen_block(n, m, p, i_start, i_start+2, j_start, j_start+2, k_start, k_start+2, offset=col)
            
            for (row, val) in u_rows:
                U_spare.append((row, col, val))
            for (row, val) in v_rows:
                V_spare.append((row, col, val))
            for (row, val) in w_rows:
                W_spare.append((row, col, val))
            col += 1
    
    rank = col
    
    # Convert to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, c, val) in U_spare:
        U[row][c] = val
    for (row, c, val) in V_spare:
        V[row][c] = val
    for (row, c, val) in W_spare:
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
