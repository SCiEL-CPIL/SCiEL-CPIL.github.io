"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _add_strassen_term(U, V, W, rank_idx, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
    """
    Adds a Strassen 2x2x2 term to the decomposition.
    a_row_off, a_col_off: offset in A (2x4)
    b_row_off, b_col_off: offset in B (4x5)
    c_row_off, c_col_off: offset in C (2x5)
    """
    # Local indices i, j, k in {0,1}
    # Global indices:
    # A: i + a_row_off, j + a_col_off
    # B: j + b_row_off, k + b_col_off
    # C: k + c_row_off, i + c_col_off
    
    # U index: (i + a_row_off) * 4 + (j + a_col_off)
    # V index: (j + b_row_off) * 5 + (k + b_col_off)
    # W index: (k + c_row_off) * 2 + (i + c_col_off)
    
    # 7 terms
    # 1. P1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[(0 + a_row_off)*4 + (0 + a_col_off)] += 1
    U[(1 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (0 + b_col_off)] += 1
    V[(1 + b_row_off)*5 + (1 + b_col_off)] += 1
    W[(0 + c_row_off)*2 + (0 + c_col_off)] += 1
    W[(1 + c_row_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # 2. P2 = (A10+A11)B00 -> C10+=1, C11-=1
    U[(1 + a_row_off)*4 + (0 + a_col_off)] += 1
    U[(1 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (0 + b_col_off)] += 1
    W[(0 + c_row_off)*2 + (1 + c_col_off)] += 1
    W[(1 + c_row_off)*2 + (1 + c_col_off)] -= 1
    rank_idx += 1
    
    # 3. P3 = A00(B10-B11) -> C01+=1, C11+=1
    U[(0 + a_row_off)*4 + (0 + a_col_off)] += 1
    V[(1 + b_row_off)*5 + (0 + b_col_off)] += 1
    V[(1 + b_row_off)*5 + (1 + b_col_off)] -= 1
    W[(0 + c_row_off)*2 + (1 + c_col_off)] += 1
    W[(1 + c_row_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # 4. P4 = A11(B01-B00) -> C00+=1, C10+=1
    U[(1 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (1 + b_col_off)] += 1
    V[(0 + b_row_off)*5 + (0 + b_col_off)] -= 1
    W[(0 + c_row_off)*2 + (0 + c_col_off)] += 1
    W[(0 + c_row_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # 5. P5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[(0 + a_row_off)*4 + (0 + a_col_off)] += 1
    U[(0 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(1 + b_row_off)*5 + (1 + b_col_off)] += 1
    W[(0 + c_row_off)*2 + (0 + c_col_off)] -= 1
    W[(0 + c_row_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # 6. P6 = (A10-A00)(B00+B01) -> C11+=1
    U[(1 + a_row_off)*4 + (0 + a_col_off)] += 1
    U[(0 + a_row_off)*4 + (0 + a_col_off)] -= 1
    V[(0 + b_row_off)*5 + (0 + b_col_off)] += 1
    V[(0 + b_row_off)*5 + (1 + b_col_off)] += 1
    W[(1 + c_row_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # 7. P7 = (A01-A11)(B10+B11) -> C00+=1
    U[(0 + a_row_off)*4 + (1 + a_col_off)] += 1
    U[(1 + a_row_off)*4 + (1 + a_col_off)] -= 1
    V[(1 + b_row_off)*5 + (0 + b_col_off)] += 1
    V[(1 + b_row_off)*5 + (1 + b_col_off)] += 1
    W[(0 + c_row_off)*2 + (0 + c_col_off)] += 1
    rank_idx += 1
    
    return rank_idx

def _add_vec_term(U, V, W, rank_idx, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
    """
    Adds a 2x2x1 vector product term.
    """
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Term 1: A00 B0 -> C0
    U[(0 + a_row_off)*4 + (0 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (b_col_off)] += 1 # Note: B col is fixed to b_col_off? No, B is 2x1 here.
    # Wait, in 2x2x1, B has 2 rows and 1 col.
    # The input B block is 2x1.
    # So B indices are (0, b_col_off) and (1, b_col_off).
    # V index: (row) * 5 + (col).
    # Row 0 of B block is global b_row_off. Col is b_col_off.
    # Row 1 of B block is global b_row_off + 1. Col is b_col_off.
    
    # Correction:
    # U: A(i, j). i in {0,1}, j in {0,1}.
    # V: B(j, k). j in {0,1}, k=0 (local).
    # W: C(k, i). k=0 (local), i in {0,1}.
    
    # Global A: (i+a_row_off, j+a_col_off)
    # Global B: (j+b_row_off, b_col_off)
    # Global C: (b_col_off, i+c_col_off)
    
    # Term 1: A00 B00 -> C00
    U[(0 + a_row_off)*4 + (0 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (b_col_off)] += 1
    W[(b_col_off)*2 + (0 + c_col_off)] += 1
    rank_idx += 1
    
    # Term 2: A01 B10 -> C00
    U[(0 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(1 + b_row_off)*5 + (b_col_off)] += 1
    W[(b_col_off)*2 + (0 + c_col_off)] += 1
    rank_idx += 1
    
    # Term 3: A10 B00 -> C10
    U[(1 + a_row_off)*4 + (0 + a_col_off)] += 1
    V[(0 + b_row_off)*5 + (b_col_off)] += 1
    W[(b_col_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    # Term 4: A11 B10 -> C10
    U[(1 + a_row_off)*4 + (1 + a_col_off)] += 1
    V[(1 + b_row_off)*5 + (b_col_off)] += 1
    W[(b_col_off)*2 + (1 + c_col_off)] += 1
    rank_idx += 1
    
    return rank_idx

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 36 decomposition
    rank = 36
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    r = 0
    
    # Block 1: A_L (rows 0,1; cols 0,1) * B_T1 (rows 0,1; cols 0,1) -> C (rows 0,1; cols 0,1)
    r = _add_strassen_term(U, V, W, r, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_L (rows 0,1; cols 0,1) * B_T2 (rows 0,1; cols 2,3) -> C (rows 0,1; cols 2,3)
    r = _add_strassen_term(U, V, W, r, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_L (rows 0,1; cols 0,1) * B_T3 (rows 0,1; col 4) -> C (rows 0,1; col 4)
    r = _add_vec_term(U, V, W, r, 0, 0, 0, 4, 0, 4)
    
    # Block 4: A_R (rows 0,1; cols 2,3) * B_B1 (rows 2,3; cols 0,1) -> C (rows 0,1; cols 0,1)
    r = _add_strassen_term(U, V, W, r, 0, 2, 2, 0, 0, 0)
    
    # Block 5: A_R (rows 0,1; cols 2,3) * B_B2 (rows 2,3; cols 2,3) -> C (rows 0,1; cols 2,3)
    r = _add_strassen_term(U, V, W, r, 0, 2, 2, 2, 0, 2)
    
    # Block 6: A_R (rows 0,1; cols 2,3) * B_B3 (rows 2,3; col 4) -> C (rows 0,1; col 4)
    r = _add_vec_term(U, V, W, r, 0, 2, 2, 4, 0, 4)
    
    # Verify r == 36
    if r != rank:
        # Fallback to schoolbook if logic error
        return _schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
