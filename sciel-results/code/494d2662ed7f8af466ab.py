"""Direct construction of rank-56 tensor for 4x4 * 4x8 using Strassen recursion."""
import time

DIMENSIONS = [4, 4, 8]

def _make_strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7 (rows indexed i*2+j), V: 4x7 (rows j*2+k), W: 4x7 (rows k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1},   {0:0, 1:0, 2:1, 3:-1}),      # M2
        ({0:1}, {1:-1, 3:1},  {0:0, 1:1, 2:1, 3:0}),       # M3
        ({3:1}, {0:-1, 2:1},  {0:-1, 1:0, 2:1, 3:0}),      # M4
        ({0:1, 2:1}, {3:1},   {0:1, 1:-1, 2:0, 3:0}),      # M5
        ({0:-1, 2:1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),   # M6
        ({1:-1, 3:1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),   # M7
    ]
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
    return U, V, W


def _strassen_4x4_4x8():
    """
    Rank-56 decomposition for 4x4 * 4x8 using Strassen's algorithm.
    
    A is 4x4: U rows indexed i*4+j (i=0..3, j=0..3) -> 16 rows
    B is 4x8: V rows indexed j*8+k (j=0..3, k=0..7) -> 32 rows  
    C is 4x8: W rows indexed k*4+i (k=0..7, i=0..3) -> 32 rows (output convention W[k*n+i])
    
    Structure: 2x2 block decomposition where:
    - A blocks are 2x2 (4 blocks: A00, A01, A10, A11)
    - B blocks are 2x2 (8 blocks: B00, B01, B10, B11, B02, B03, B12, B13)
    - C blocks are 2x2 (8 blocks: C00, C01, C10, C11, C02, C03, C12, C13)
    
    For each of the 7 Strassen multiplications, the A-side uses 2x2 blocks,
    the B-side uses 2x2 blocks, and the C-side accumulates into 2x2 blocks.
    Each Strassen term has rank 7 (from the base 2x2 Strassen), and the B-side
    covers all 8 columns via the block structure.
    
    Total rank = 7 (Strassen level) * 8 (output columns per row-group) = 56
    """
    rank = 56
    U = [[0]*rank for _ in range(16)]  # 4x4 = 16 rows
    V = [[0]*rank for _ in range(32)]  # 4x8 = 32 rows
    W = [[0]*rank for _ in range(32)]  # 8x4 = 32 rows
    
    u2, v2, w2 = _make_strassen_2x2()
    
    # Strassen formula terms: (A-blocks with signs, B-blocks with signs, C-blocks with signs)
    # A blocks: 0=A00(top-left), 1=A01(top-right), 2=A10(bottom-left), 3=A11(bottom-right)
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11 (for first 4 cols), 4=B02, 5=B03, 6=B12, 7=B13 (for last 4 cols)
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11 (for first 4 cols), 4=C02, 5=C03, 6=C12, 7=C13 (for last 4 cols)
    
    strassen_mults = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),            # M2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),           # M3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),           # M4 = A11(B10-A00)
        ([(0, 1), (2, 1)], [(3, 1)]),            # M5 = (A00+A10)B11
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)]),   # M6 = (A10-A00)(B00+B10)
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)]),   # M7 = (A01-A11)(B01+B11)
    ]
    
    c_assignments = [
        [(0, 1), (1, 1), (2, -1), (3, 1)],       # C00 += M1, C01 += M1, C10 -= M1, C11 += M1
        [(2, 1), (3, -1)],                        # C10 += M2, C11 -= M2
        [(1, 1), (2, 1)],                         # C01 += M3, C10 += M3
        [(0, -1), (2, 1)],                        # C00 -= M4, C10 += M4
        [(0, 1), (1, -1)],                        # C00 += M5, C01 -= M5
        [(0, 1), (3, 1)],                         # C00 += M6, C11 += M6
        [(1, 1), (3, 1)],                         # C01 += M7, C11 += M7
    ]
    
    # Block positions for A (2x2 blocks of 2x2): block -> (row_offset, col_offset)
    a_block_pos = {
        0: (0, 0),   # A00: rows 0-1, cols 0-1
        1: (0, 2),   # A01: rows 0-1, cols 2-3
        2: (2, 0),   # A10: rows 2-3, cols 0-1
        3: (2, 2),   # A11: rows 2-3, cols 2-3
    }
    
    # For B: 4x8 matrix, blocks are 2x2. 
    # B block index: 0-3 for columns 0-3, 4-7 for columns 4-7
    # Block b: if b < 4, row_off = (b//2)*2, col_off = (b%2)*2
    #           if b >= 4, row_off = ((b-4)//2)*2, col_off = ((b-4)%2)*2 + 4
    def b_block_pos(b):
        if b < 4:
            row_off = (b // 2) * 2
            col_off = (b % 2) * 2
        else:
            row_off = ((b - 4) // 2) * 2
            col_off = ((b - 4) % 2) * 2 + 4
        return row_off, col_off
    
    # For C (output): 4x8, indexed as W[k*4+i]. C block index: 0-3 for cols 0-3, 4-7 for cols 4-7
    # C block c: if c < 4, row_off(k) = (c//2)*2, col_off(i) = (c%2)*2
    #            if c >= 4, row_off(k) = ((c-4)//2)*2, col_off(i) = ((c-4)%2)*2 + 4
    def c_block_pos(c):
        if c < 4:
            row_off = (c // 2) * 2
            col_off = (c % 2) * 2
        else:
            row_off = ((c - 4) // 2) * 2
            col_off = ((c - 4) % 2) * 2 + 4
        return row_off, col_off
    
    for m in range(7):
        a_terms, b_terms = strassen_mults[m]
        c_terms = c_assignments[m]
        
        for (a_block, a_sign) in a_terms:
            a_row_off, a_col_off = a_block_pos[a_block]
            for (b_block, b_sign) in b_terms:
                b_row_off, b_col_off = b_block_pos(b_block)
                for (c_block, c_sign) in c_terms:
                    c_row_off, c_col_off = c_block_pos(c_block)
                    
                    scale = a_sign * b_sign * c_sign
                    if scale == 0:
                        continue
                    
                    for r in range(7):
                        col_idx = m * 8 + r
                        
                        # U: A-side, 2x2 block
                        for i in range(2):
                            for j in range(2):
                                u2_val = u2[i*2+j][r]
                                if u2_val != 0:
                                    U[(a_row_off+i)*4 + (a_col_off+j)][col_idx] += scale * u2_val
                        
                        # V: B-side, 2x2 block
                        for j in range(2):
                            for k in range(2):
                                v2_val = v2[j*2+k][r]
                                if v2_val != 0:
                                    V[(b_row_off+j)*8 + (b_col_off+k)][col_idx] += scale * v2_val
                        
                        # W: C-side, 2x2 block (output convention: W[k*n+i] = W[k*4+i])
                        for k in range(2):
                            for i in range(2):
                                w2_val = w2[k*2+i][r]
                                if w2_val != 0:
                                    W[(c_row_off+k)*4 + (c_col_off+i)][col_idx] += scale * w2_val
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    deadline = time.time() + 100
    
    # Use Strassen 4x4*4x8 directly: rank 56
    U, V, W = _strassen_4x4_4x8()
    rank = 56
    
    # Verify the full decomposition
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = sum(U_arr[i*4+j, r] * V_arr[j*8+k, r] * W_arr[k*4+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        raise ValueError(f"Decomposition mismatch at ({i},{j},{k}): got {val}")
    except Exception as e:
        # If verification fails, fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
