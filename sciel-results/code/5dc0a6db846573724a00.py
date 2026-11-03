"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-66 decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy: Split A (3x4) into row blocks [1, 2] and col blocks [2, 2].
    Split B (4x6) into row blocks [2, 2] and col blocks [2, 2, 2].
    
    For each pair (A_block, B_block) sharing the middle dimension:
    - If A_block is 1x2: use schoolbook rank 4 for 1x2*2x2
    - If A_block is 2x2: use Strassen rank 7 for 2x2*2x2
    
    Total: 6 blocks of rank 4 + 6 blocks of rank 7 = 24 + 42 = 66
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll collect all rank-1 terms as (u_coeff, v_coeff, w_coeff) triples
    # where u_coeff is a dict mapping (i,j) -> value
    # v_coeff is a dict mapping (j,k) -> value  
    # w_coeff is a dict mapping (k,i) -> value
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term with sparse coefficient dicts."""
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_1x2x2(a_row, a_col_start, b_row_start, b_col_start):
        """
        Schoolbook decomposition for A[a_row, a_col_start:a_col_start+2] * B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        This is a 1x2 * 2x2 block, giving a 1x2 result.
        Rank = 1*2*2 = 4
        """
        # A has 1 row (a_row), 2 cols (a_col_start, a_col_start+1)
        # B has 2 rows (b_row_start, b_row_start+1), 2 cols (b_col_start, b_col_start+1)
        # C has 1 row (a_row), 2 cols (b_col_start, b_col_start+1)
        
        # 4 rank-1 terms:
        for j in range(2):
            for k in range(2):
                u = {(a_row, a_col_start + j): 1}
                v = {(b_row_start + j, b_col_start + k): 1}
                w = {(b_col_start + k, a_row): 1}
                add_term(u, v, w)
    
    def strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start):
        """
        Strassen rank-7 decomposition for 2x2 * 2x2 block.
        A is 2x2 at rows a_row_start:a_row_start+2, cols a_col_start:a_col_start+2
        B is 2x2 at rows b_row_start:b_row_start+2, cols b_col_start:b_col_start+2
        C is 2x2 at rows a_row_start:a_row_start+2, cols b_col_start:b_col_start+2
        
        The 7 Strassen products and their contributions to C:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B00-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        Each M_r is a product of a linear combo of A entries and a linear combo of B entries.
        In the tensor decomposition, each M_r corresponds to a rank-1 term where:
        - u vector has the A coefficients
        - v vector has the B coefficients
        - w vector has the C contribution coefficients
        """
        # Define indices
        a00 = (a_row_start, a_col_start)
        a01 = (a_row_start, a_col_start + 1)
        a10 = (a_row_start + 1, a_col_start)
        a11 = (a_row_start + 1, a_col_start + 1)
        
        b00 = (b_row_start, b_col_start)
        b01 = (b_row_start, b_col_start + 1)
        b10 = (b_row_start + 1, b_col_start)
        b11 = (b_row_start + 1, b_col_start + 1)
        
        c00 = (a_row_start, b_col_start)
        c01 = (a_row_start, b_col_start + 1)
        c10 = (a_row_start + 1, b_col_start)
        c11 = (a_row_start + 1, b_col_start + 1)
        
        # M1 = (A00+A11)(B00+B11), contributes to C00 (+1) and C11 (+1)
        u1 = {a00: 1, a11: 1}
        v1 = {b00: 1, b11: 1}
        w1 = {c00: 1, c11: 1}
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00, contributes to C10 (+1) and C11 (-1)
        u2 = {a10: 1, a11: 1}
        v2 = {b00: 1}
        w2 = {c10: 1, c11: -1}
        add_term(u2, v2, w2)
        
        # M3 = A00(B10-B11), contributes to C01 (+1) and C11 (+1)
        u3 = {a00: 1}
        v3 = {b10: 1, b11: -1}
        w3 = {c01: 1, c11: 1}
        add_term(u3, v3, w3)
        
        # M4 = A11(B00-B10), contributes to C00 (+1) and C10 (+1)
        u4 = {a11: 1}
        v4 = {b00: 1, b10: -1}
        w4 = {c00: 1, c10: 1}
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11, contributes to C00 (-1) and C01 (+1)
        u5 = {a00: 1, a01: 1}
        v5 = {b11: 1}
        w5 = {c00: -1, c01: 1}
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01), contributes to C11 (+1)
        u6 = {a10: 1, a00: -1}
        v6 = {b00: 1, b01: 1}
        w6 = {c11: 1}
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11), contributes to C00 (+1)
        u7 = {a01: 1, a11: -1}
        v7 = {b10: 1, b11: 1}
        w7 = {c00: 1}
        add_term(u7, v7, w7)
    
    # Now build the block decomposition
    # A row splits: [0:1], [1:3]
    # A col splits: [0:2], [2:4]
    # B row splits: [0:2], [2:4]
    # B col splits: [0:2], [2:4], [4:6]
    
    a_row_splits = [(0, 1), (1, 3)]
    a_col_splits = [(0, 2), (2, 4)]
    b_row_splits = [(0, 2), (2, 4)]
    b_col_splits = [(0, 2), (2, 4), (4, 6)]
    
    for a_row_start, a_row_end in a_row_splits:
        a_row_count = a_row_end - a_row_start
        for a_col_start, a_col_end in a_col_splits:
            for b_col_start, b_col_end in b_col_splits:
                # The middle dimension is split as [0:2], [2:4]
                # A block uses cols a_col_start:a_col_end
                # B block uses rows that match: b_row_start = a_col_start, b_row_end = a_col_end
                b_row_start = a_col_start
                b_row_end = a_col_end
                
                if a_row_count == 1:
                    # 1x2 * 2x2 block
                    schoolbook_1x2x2(a_row_start, a_col_start, b_row_start, b_col_start)
                else:
                    # 2x2 * 2x2 block
                    strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start)
    
    # Check time
    if time.time() > deadline:
        # Fall back to schoolbook
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
    
    # Now convert the list of terms into U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
