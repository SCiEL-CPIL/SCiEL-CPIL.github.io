"""Rewritten program for matrix multiplication tensor decomposition M_{3,5,7}."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Target rank: 99
    # Decomposition:
    # M_{3,5,7} = M_{3,2,7} + M_{3,3,7}
    # M_{3,2,7} = M_{1,2,7} (Rank 14) + M_{2,2,7} (Rank 25)
    # M_{3,3,7} = M_{1,3,7} (Rank 21) + M_{2,3,7} (Rank 39)
    # M_{2,3,7} = M_{2,1,7} (Rank 14) + M_{2,2,7} (Rank 25)
    # M_{2,2,7} = M_{2,2,2} (Rank 7) + M_{2,2,2} (Rank 7) + M_{2,2,3} (Rank 11)
    # M_{2,2,3} = M_{2,2,1} (Rank 4) + M_{2,2,2} (Rank 7)
    
    total_rank = 99
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    col = 0
    
    def add_schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        for i in a_rows:
            for j in a_cols:
                for k in b_cols:
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows: [r1, r2], a_cols: [c1, c2]
        # b_rows: [r1, r2], b_cols: [c1, c2]
        # c_rows: [r1, r2], c_cols: [c1, c2]
        
        r1, r2 = a_rows
        c1, c2 = a_cols
        br1, br2 = b_rows
        bc1, bc2 = b_cols
        cr1, cr2 = c_rows
        cc1, cc2 = c_cols
        
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        
        # Term 1
        U[r1 * m + c1, col] = 1
        U[r2 * m + c2, col] = 1
        V[br1 * p + bc1, col] = 1
        V[br2 * p + bc2, col] = 1
        W[cc1 * n + cr1, col] = 1
        W[cc2 * n + cr2, col] = 1
        col += 1
        
        # Term 2
        U[r2 * m + c1, col] = 1
        U[r2 * m + c2, col] = 1
        V[br1 * p + bc1, col] = 1
        W[cc1 * n + cr2, col] = 1
        W[cc2 * n + cr2, col] = 1
        col += 1
        
        # Term 3
        U[r1 * m + c1, col] = 1
        V[br1 * p + bc2, col] = 1
        V[br2 * p + bc2, col] = -1
        W[cc2 * n + cr1, col] = 1
        col += 1
        
        # Term 4
        U[r2 * m + c2, col] = 1
        V[br2 * p + bc1, col] = 1
        V[br1 * p + bc1, col] = -1
        W[cc1 * n + cr1, col] = 1
        W[cc1 * n + cr2, col] = 1
        col += 1
        
        # Term 5
        U[r1 * m + c1, col] = 1
        U[r1 * m + c2, col] = 1
        V[br2 * p + bc2, col] = 1
        W[cc1 * n + cr1, col] = -1
        W[cc2 * n + cr1, col] = 1
        col += 1
        
        # Term 6
        U[r2 * m + c1, col] = 1
        U[r1 * m + c1, col] = -1
        V[br1 * p + bc1, col] = 1
        V[br1 * p + bc2, col] = 1
        W[cc2 * n + cr2, col] = 1
        col += 1
        
        # Term 7
        U[r1 * m + c2, col] = 1
        U[r2 * m + c2, col] = -1
        V[br2 * p + bc1, col] = 1
        V[br2 * p + bc2, col] = 1
        W[cc1 * n + cr1, col] = 1
        W[cc2 * n + cr2, col] = 1
        col += 1

    # Block 1: M_{1,2,7} (A row 2, B rows 0-1, C cols 0-6)
    # A rows: [2], A cols: [0, 1]
    # B rows: [0, 1], B cols: [0..6]
    # C rows: [2], C cols: [0..6]
    add_schoolbook_block([2], [0, 1], [0, 1], list(range(7)), [2], list(range(7)))
    
    # Block 2: M_{2,2,7} (A rows 0-1, B rows 0-1, C rows 0-1, C cols 0-6)
    # A rows: [0, 1], A cols: [0, 1]
    # B rows: [0, 1]
    # C rows: [0, 1]
    
    # Sub-block 2.1: M_{2,2,2} (C cols 0-1)
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Sub-block 2.2: M_{2,2,2} (C cols 2-3)
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Sub-block 2.3: M_{2,2,3} (C cols 4-6)
    # 2.3.1: M_{2,2,1} (C col 4)
    add_schoolbook_block([0, 1], [0, 1], [0, 1], [4], [0, 1], [4])
    
    # 2.3.2: M_{2,2,2} (C cols 5-6)
    add_strassen_block([0, 1], [0, 1], [0, 1], [5, 6], [0, 1], [5, 6])
    
    # Block 3: M_{1,3,7} (A row 2, B rows 2-4, C cols 0-6)
    add_schoolbook_block([2], [2, 3, 4], [2, 3, 4], list(range(7)), [2], list(range(7)))
    
    # Block 4: M_{2,3,7} (A rows 0-1, B rows 2-4, C rows 0-1, C cols 0-6)
    # 4.1: M_{2,1,7} (B row 2)
    add_schoolbook_block([0, 1], [2], [2], list(range(7)), [0, 1], list(range(7)))
    
    # 4.2: M_{2,2,7} (B rows 3-4)
    # A rows: [0, 1], A cols: [3, 4]
    # B rows: [3, 4]
    # C rows: [0, 1]
    
    # Sub-block 4.2.1: M_{2,2,2} (C cols 0-1)
    add_strassen_block([0, 1], [3, 4], [3, 4], [0, 1], [0, 1], [0, 1])
    
    # Sub-block 4.2.2: M_{2,2,2} (C cols 2-3)
    add_strassen_block([0, 1], [3, 4], [3, 4], [2, 3], [0, 1], [2, 3])
    
    # Sub-block 4.2.3: M_{2,2,3} (C cols 4-6)
    # 4.2.3.1: M_{2,2,1} (C col 4)
    add_schoolbook_block([0, 1], [3, 4], [3, 4], [4], [0, 1], [4])
    
    # 4.2.3.2: M_{2,2,2} (C cols 5-6)
    add_strassen_block([0, 1], [3, 4], [3, 4], [5, 6], [0, 1], [5, 6])
    
    if col != total_rank:
        # Fallback to schoolbook if construction failed
        def schoolbook():
            r = n * m * p
            U = np.zeros((n * m, r), dtype=np.int64)
            V = np.zeros((m * p, r), dtype=np.int64)
            W = np.zeros((p * n, r), dtype=np.int64)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                        W[k * n + i, c] = 1
                        c += 1
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        return schoolbook()

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
