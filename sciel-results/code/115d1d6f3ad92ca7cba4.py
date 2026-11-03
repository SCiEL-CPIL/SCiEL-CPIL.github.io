import numpy as np
import time

T0 = time.time()
DEADLINE = 150.0
SAFETY = 10.0

# Precompute the 28-term Strassen decomposition for 2x4x4
def _build_strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 coefficients
    # For 2x2 matrices A, B, C:
    # A = [[a00, a01], [a10, a11]]  -> flat index: row*m+col
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    strassen = [
        # (A coeffs {flat_idx: coef}, B coeffs, C coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def emit_block(a_row, b_col, c_row, c_col, off):
        """
        a_row: 0 or 2 (row offset in A's 4 cols... wait, A is 2x4, so rows 0,1)
        Actually: A is n x m = 2 x 4.
        We split A into 2x2 blocks: A[0:2, 0:2] and A[0:2, 2:4]
        So a_row is the row block (0), and we need col block.
        
        Let me re-examine the structure:
        A: 2 x 4 -> rows {0,1}, cols {0,1,2,3}
        B: 4 x 4 -> rows {0,1,2,3}, cols {0,1,2,3}
        C: 2 x 4 -> rows {0,1}, cols {0,1,2,3}
        
        We use 4 2x2x2 Strassen multiplications:
        Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        Block 2: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
        Block 3: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2] (added)
        Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (added)
        
        Wait, that's not right. Let me think again.
        
        A (2x4) * B (4x4) = C (2x4)
        C[i][j] = sum_{k=0}^{3} A[i][k] * B[k][j]
        
        We can split k into {0,1} and {2,3}:
        C[i][j] = sum_{k=0}^{1} A[i][k]*B[k][j] + sum_{k=2}^{3} A[i][k]*B[k][j]
        
        For each pair of k-values and j-values, we have a 2x2x2 multiplication.
        A is 2x2 (rows i=0,1; cols k), B is 2x2 (rows k; cols j), C is 2x2 (rows i; cols j).
        
        So we have 4 blocks:
        Block (ak, bj) where ak in {0,1} (k offset 0 or 2), bj in {0,1} (j offset 0 or 2):
        - Block (0,0): A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        - Block (0,1): A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        - Block (1,0): A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        - Block (1,1): A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        """
        a_col_off = a_row * 2  # a_row is k-block: 0 or 1, so cols 0:2 or 2:4
        b_row_off = a_row * 2  # B row offset = k offset
        b_col_off = b_col * 2  # b_col is j-block: 0 or 1, so cols 0:2 or 2:4
        c_col_off = b_col * 2  # C col offset = j offset
        
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                # li is flat index in 2x2 A block: row i, col (k local)
                r_i, c_k = li // 2, li % 2
                U[r_i * m + a_col_off + c_k][col] += c
            for li, c in vc.items():
                # li is flat index in 2x2 B block: row (k local), col j
                r_k, c_j = li // 2, li % 2
                V[(b_row_off + r_k) * p + b_col_off + c_j][col] += c
            for li, c in wc.items():
                # li is flat index in 2x2 C block: row i, col (j local)
                r_i, c_j = li // 2, li % 2
                W[(c_col_off + c_j) * n + r_i][col] += c
    
    # 4 blocks
    emit_block(0, 0, 0, 0, 0)   # k=0:2, j=0:2
    emit_block(0, 1, 0, 0, 7)   # k=0:2, j=2:4
    emit_block(1, 0, 0, 0, 14)  # k=2:4, j=0:2
    emit_block(1, 1, 0, 0, 21)  # k=2:4, j=2:4
    
    return U, V, W

def construct():
    U, V, W = _build_strassen_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
