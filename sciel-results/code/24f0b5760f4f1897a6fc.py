"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _add_strassen_2x2x2(U, V, W, rank_idx, a_r, a_c, b_r, b_c, c_r, c_c):
    """
    Adds a Strassen 2x2x2 term.
    A block at (a_r, a_c) size 2x2.
    B block at (b_r, b_c) size 2x2.
    C block at (c_r, c_c) size 2x2.
    """
    # Indices in U (n*m, rank): n=2, m=4.
    # A[i, j] -> U[i*4 + j]
    # B[i, j] -> V[i*5 + j]
    # C[i, j] -> W[i*2 + j]
    
    # 1. M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[(a_r+0)*4 + (a_c+0)] += 1
    U[(a_r+1)*4 + (a_c+1)] += 1
    V[(b_r+0)*5 + (b_c+0)] += 1
    V[(b_r+1)*5 + (b_c+1)] += 1
    W[(c_r+0)*2 + (c_c+0)] += 1
    W[(c_r+1)*2 + (c_c+1)] += 1
    rank_idx += 1
    
    # 2. M2 = (A10+A11)B00 -> C10+=1, C11-=1
    U[(a_r+1)*4 + (a_c+0)] += 1
    U[(a_r+1)*4 + (a_c+1)] += 1
    V[(b_r+0)*5 + (b_c+0)] += 1
    W[(c_r+1)*2 + (c_c+0)] += 1
    W[(c_r+1)*2 + (c_c+1)] -= 1
    rank_idx += 1
    
    # 3. M3 = A00(B10-B11) -> C01+=1, C11+=1
    U[(a_r+0)*4 + (a_c+0)] += 1
    V[(b_r+1)*5 + (b_c+0)] += 1
    V[(b_r+1)*5 + (b_c+1)] -= 1
    W[(c_r+0)*2 + (c_c+1)] += 1
    W[(c_r+1)*2 + (c_c+1)] += 1
    rank_idx += 1
    
    # 4. M4 = A11(B01-B00) -> C00+=1, C10+=1
    U[(a_r+1)*4 + (a_c+1)] += 1
    V[(b_r+0)*5 + (b_c+1)] += 1
    V[(b_r+0)*5 + (b_c+0)] -= 1
    W[(c_r+0)*2 + (c_c+0)] += 1
    W[(c_r+1)*2 + (c_c+0)] += 1
    rank_idx += 1
    
    # 5. M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[(a_r+0)*4 + (a_c+0)] += 1
    U[(a_r+0)*4 + (a_c+1)] += 1
    V[(b_r+1)*5 + (b_c+1)] += 1
    W[(c_r+0)*2 + (c_c+0)] -= 1
    W[(c_r+0)*2 + (c_c+1)] += 1
    rank_idx += 1
    
    # 6. M6 = (A10-A00)(B00+B01) -> C11+=1
    U[(a_r+1)*4 + (a_c+0)] += 1
    U[(a_r+0)*4 + (a_c+0)] -= 1
    V[(b_r+0)*5 + (b_c+0)] += 1
    V[(b_r+0)*5 + (b_c+1)] += 1
    W[(c_r+1)*2 + (c_c+1)] += 1
    rank_idx += 1
    
    # 7. M7 = (A01-A11)(B10+B11) -> C00+=1
    U[(a_r+0)*4 + (a_c+1)] += 1
    U[(a_r+1)*4 + (a_c+1)] -= 1
    V[(b_r+1)*5 + (b_c+0)] += 1
    V[(b_r+1)*5 + (b_c+1)] += 1
    W[(c_r+0)*2 + (c_c+0)] += 1
    rank_idx += 1
    
    return rank_idx

def _add_schoolbook_2x2x3(U, V, W, rank_idx, a_r, a_c, b_r, b_c, c_r, c_c):
    """
    Adds schoolbook terms for 2x2 A * 2x3 B -> 2x3 C.
    A is 2x2 at (a_r, a_c).
    B is 2x3 at (b_r, b_c).
    C is 2x3 at (c_r, c_c).
    """
    # C[i, k] = sum_j A[i, j] B[j, k]
    for i in range(2):
        for k in range(3):
            for j in range(2):
                U[(a_r+i)*4 + (a_c+j)] += 1
                V[(b_r+j)*5 + (b_c+k)] += 1
                W[(c_r+i)*2 + (c_c+k)] += 1
                rank_idx += 1
    return rank_idx

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank: 38
    # Decomposition:
    # C = A_L B_T + A_R B_B
    # A_L: cols 0,1 of A. B_T: rows 0,1 of B.
    # A_R: cols 2,3 of A. B_B: rows 2,3 of B.
    
    # Each 2x2 * 2x5 block is split into 2x2 * 2x2 (Strassen, rank 7) and 2x2 * 2x3 (Schoolbook, rank 12).
    # Total rank = (7 + 12) * 2 = 38.
    
    rank = 38
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    r = 0
    
    # Block 1: A_L (rows 0,1; cols 0,1) * B_T (rows 0,1; cols 0,4)
    # Sub-block 1a: B_T cols 0,1 (2x2)
    r = _add_strassen_2x2x2(U, V, W, r, a_r=0, a_c=0, b_r=0, b_c=0, c_r=0, c_c=0)
    # Sub-block 1b: B_T cols 2,3,4 (2x3)
    r = _add_schoolbook_2x2x3(U, V, W, r, a_r=0, a_c=0, b_r=0, b_c=2, c_r=0, c_c=2)
    
    # Block 2: A_R (rows 0,1; cols 2,3) * B_B (rows 2,3; cols 0,4)
    # Sub-block 2a: B_B cols 0,1 (2x2)
    r = _add_strassen_2x2x2(U, V, W, r, a_r=0, a_c=2, b_r=2, b_c=0, c_r=0, c_c=0)
    # Sub-block 2b: B_B cols 2,3,4 (2x3)
    r = _add_schoolbook_2x2x3(U, V, W, r, a_r=0, a_c=2, b_r=2, b_c=2, c_r=0, c_c=2)
    
    if r != rank:
        # Fallback to schoolbook rank 40 if logic error (should not happen)
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
