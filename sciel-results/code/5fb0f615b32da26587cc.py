import numpy as np
import time

T0 = time.time()
DEADLINE = 150.0
SAFETY = 10.0

def _build_strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 coefficients
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def emit_block(a_col_off, b_row_off, b_col_off, c_col_off, off):
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r_i, c_k = li // 2, li % 2
                U[r_i * m + a_col_off + c_k][col] += c
            for li, c in vc.items():
                r_k, c_j = li // 2, li % 2
                V[(b_row_off + r_k) * p + b_col_off + c_j][col] += c
            for li, c in wc.items():
                r_i, c_j = li // 2, li % 2
                W[(c_col_off + c_j) * n + r_i][col] += c
    
    # 4 blocks
    emit_block(0, 0, 0, 0, 0)   # k=0:2, j=0:2
    emit_block(0, 0, 2, 2, 7)   # k=0:2, j=2:4
    emit_block(2, 2, 0, 0, 14)  # k=2:4, j=0:2
    emit_block(2, 2, 2, 2, 21)  # k=2:4, j=2:4
    
    return U, V, W

def construct():
    U, V, W = _build_strassen_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
