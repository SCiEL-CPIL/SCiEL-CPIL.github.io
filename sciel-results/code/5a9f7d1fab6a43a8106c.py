"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_terms(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, U, V, W, col_start):
    """
    Add Strassen terms for computing C[c_rows,c_cols] += A[a_rows,a_cols] * B[b_rows,b_cols]
    where A is 2x2, B is 2x2, C is 2x2.
    Returns new column index.
    """
    col = col_start
    a0, a1 = a_rows
    b0, b1 = a_cols  # Wait, I need to be more careful with indices
    # A indices: (a_rows[0], a_cols[0]), (a_rows[0], a_cols[1]), (a_rows[1], a_cols[0]), (a_rows[1], a_cols[1])
    # B indices: (b_rows[0], b_cols[0]), etc.
    # C indices: (c_rows[0], c_cols[0]), etc.
    
    # U index for A[i,j] is i*5 + j (since m=5)
    # V index for B[j,k] is j*5 + k (since p=5)
    # W index for C[k,i] is k*4 + i (since n=4)
    
    ar0, ar1 = a_rows
    ac0, ac1 = a_cols
    br0, br1 = b_rows
    bc0, bc1 = b_cols
    cr0, cr1 = c_rows
    cc0, cc1 = c_cols
    
    # M1 = (A[ar0,ac0] + A[ar1,ac1]) * (B[br0,bc0] + B[br1,bc1])
    U[ar0*5+ac0][col] = 1
    U[ar1*5+ac1][col] = 1
    V[br0*5+bc0][col] = 1
    V[br1*5+bc1][col] = 1
    # C contribution: C[cr0,cc0] += 1, C[cr0,cc1] += 1, C[cr1,cc0] += 1, C[cr1,cc1] += 1
    # W[k*n+i] where C[i,k] gets the value
    W[cc0*4+cr0][col] = 1
    W[cc1*4+cr0][col] = 1
    W[cc0*4+cr1][col] = 1
    W[cc1*4+cr1][col] = 1
    col += 1
    
    # M2 = (A[ar1,ac0] + A[ar1,ac1]) * B[br0,bc0]
    U[ar1*5+ac0][col] = 1
    U[ar1*5+ac1][col] = 1
    V[br0*5+bc0][col] = 1
    W[cc0*4+cr1][col] = 1
    W[cc1*4+cr1][col] = 1
    col += 1
    
    # M3 = A[ar0,ac0] * (B[br1,bc0] - B[br1,bc1])
    U[ar0*5+ac0][col] = 1
    V[br1*5+bc0][col] = 1
    V[br1*5+bc1][col] = -1
    W[cc0*4+cr0][col] = 1
    W[cc1*4+cr0][col] = 1
    col += 1
    
    # M4 = A[ar1,ac1] * (B[br0,bc1] - B[br0,bc0])
    U[ar1*5+ac1][col] = 1
    V[br0*5+bc1][col] = 1
    V[br0*5+bc0][col] = -1
    W[cc0*4+cr1][col] = 1
    W[cc1*4+cr1][col] = 1
    col += 1
    
    # M5 = (A[ar0,ac0] + A[ar0,ac1]) * B[br1,bc1]
    U[ar0*5+ac0][col] = 1
    U[ar0*5+ac1][col] = 1
    V[br1*5+bc1][col] = 1
    W[cc0*4+cr0][col] = -1
    W[cc1*4+cr0][col] = -1
    col += 1
    
    # M6 = (A[ar1,ac0] - A[ar0,ac0]) * (B[br0,bc0] + B[br0,bc1])
    U[ar1*5+ac0][col] = 1
    U[ar0*5+ac0][col] = -1
    V[br0*5+bc0][col] = 1
    V[br0*5+bc1][col] = 1
    W[cc1*4+cr1][col] = 1
    col += 1
    
    # M7 = (A[ar0,ac1] - A[ar1,ac1]) * (B[br1,bc0] + B[br1,bc1])
    U[ar0*5+ac1][col] = 1
    U[ar1*5+ac1][col] = -1
    V[br1*5+bc0][col] = 1
    V[br1*5+bc1][col] = 1
    W[cc0*4+cr0][col] = 1
    W[cc1*4+cr1][col] = 1
    col += 1
    
    return col


def _schoolbook_2x2x2_terms(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, U, V, W, col_start):
    """Add schoolbook terms for C += A*B where all are 2x2."""
    col = col_start
    for i_idx, ai in enumerate(a_rows):
        for j_idx, aj in enumerate(a_cols):
            for k_idx, bk in enumerate(b_cols):
                # C[c_rows[i_idx], c_cols[k_idx]] += A[ai, aj] * B[b_rows[j_idx], bk]
                U[ai*5+aj][col] = 1
                V[b_rows[j_idx]*5+bk][col] = 1
                W[bk*4+c_rows[i_idx]][col] = 1
                col += 1
    return col


def _schoolbook_2x1x2_terms(a_rows, a_col, b_row, b_cols, c_rows, c_cols, U, V, W, col_start):
    """Add schoolbook terms for C (2x2) += A (2x1) * B (1x2)."""
    col = col_start
    for i_idx, ai in enumerate(a_rows):
        for k_idx, bk in enumerate(b_cols):
            U[ai*5+a_col][col] = 1
            V[b_row*5+bk][col] = 1
            W[bk*4+c_rows[i_idx]][col] = 1
            col += 1
    return col


def _schoolbook_2x2x1_terms(a_rows, a_cols, b_rows, b_col, c_rows, c_col, U, V, W, col_start):
    """Add schoolbook terms for C (2x1) += A (2x2) * B (2x1)."""
    col = col_start
    for i_idx, ai in enumerate(a_rows):
        for j_idx, aj in enumerate(a_cols):
            U[ai*5+aj][col] = 1
            V[b_rows[j_idx]*5+b_col][col] = 1
            W[b_col*4+c_rows[i_idx]][col] = 1
            col += 1
    return col


def _schoolbook_2x1x1_terms(a_rows, a_col, b_row, b_col, c_rows, c_col, U, V, W, col_start):
    """Add schoolbook terms for C (2x1) += A (2x1) * B (1x1)."""
    col = col_start
    for i_idx, ai in enumerate(a_rows):
        U[ai*5+a_col][col] = 1
        V[b_row*5+b_col][col] = 1
        W[b_col*4+c_rows[i_idx]][col] = 1
        col += 1
    return col


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block structure:
    # A rows: {0,1}, {2,3}
    # A cols / B rows: {0,1}, {2,3}, {4}
    # B cols / C cols: {0,1}, {2,3}, {4}
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2] + A[0:2,4:5]*B[4:5,0:2]
    col = _strassen_2x2_terms([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], U, V, W, col)
    col = _strassen_2x2_terms([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], U, V, W, col)
    col = _schoolbook_2x1x2_terms([0,1], 4, 4, [0,1], [0,1], [0,1], U, V, W, col)
    
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4] + A[0:2,4:5]*B[4:5,2:4]
    col = _strassen_2x2_terms([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], U, V, W, col)
    col = _strassen_2x2_terms([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], U, V, W, col)
    col = _schoolbook_2x1x2_terms([0,1], 4, 4, [2,3], [0,1], [2,3], U, V, W, col)
    
    # C[0:2, 4:5] = A[0:2,0:2]*B[0:2,4:5] + A[0:2,2:4]*B[2:4,4:5] + A[0:2,4:5]*B[4:5,4:5]
    col = _schoolbook_2x2x1_terms([0,1], [0,1], [0,1], 4, [0,1], 4, U, V, W, col)
    col = _schoolbook_2x2x1_terms([0,1], [2,3], [2,3], 4, [0,1], 4, U, V, W, col)
    col = _schoolbook_2x1x1_terms([0,1], 4, 4, 4, [0,1], 4, U, V, W, col)
    
    # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2] + A[2:4,4:5]*B[4:5,0:2]
    col = _strassen_2x2_terms([2,3], [0,1], [0,1], [0,1], [2,3], [0,1], U, V, W, col)
    col = _strassen_2x2_terms([2,3], [2,3], [2,3], [0,1], [2,3], [0,1], U, V, W, col)
    col = _schoolbook_2x1x2_terms([2,3], 4, 4, [0,1], [2,3], [0,1], U, V, W, col)
    
    # C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4] + A[2:4,4:5]*B[4:5,2:4]
    col = _strassen_2x2_terms([2,3], [0,1], [0,1], [2,3], [2,3], [2,3], U, V, W, col)
    col = _strassen_2x2_terms([2,3], [2,3], [2,3], [2,3], [2,3], [2,3], U, V, W, col)
    col = _schoolbook_2x1x2_terms([2,3], 4, 4, [2,3], [2,3], [2,3], U, V, W, col)
    
    # C[2:4, 4:5] = A[2:4,0:2]*B[0:2,4:5] + A[2:4,2:4]*B[2:4,4:5] + A[2:4,4:5]*B[4:5,4:5]
    col = _schoolbook_2x2x1_terms([2,3], [0,1], [0,1], 4, [2,3], 4, U, V, W, col)
    col = _schoolbook_2x2x1_terms([2,3], [2,3], [2,3], 4, [2,3], 4, U, V, W, col)
    col = _schoolbook_2x1x1_terms([2,3], 4, 4, 4, [2,3], 4, U, V, W, col)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
