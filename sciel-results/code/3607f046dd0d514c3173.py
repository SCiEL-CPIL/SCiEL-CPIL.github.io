import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DL = 130.0

    n, m, p = DIMENSIONS
    nb = p // 2  # 7 blocks of 2x2

    # Strassen base matrices for 2x2 * 2x2 -> 2x2
    # U_rows: 4x7 (rows: a,b,c,d; cols: M1..M7)
    # V_base: 7x4 (rows: M1..M7; cols: e,f,g,h)
    # W_base: 7x4 (rows: M1..M7; cols: C11,C12,C21,C22)
    Sr = np.array([[1,0,1,0,1,-1,0],
                   [0,0,0,0,1,0,1],
                   [0,1,0,0,0,1,0],
                   [1,1,0,1,0,0,-1]], dtype=np.int64)
    Vb = np.array([[1,0,0,1],
                   [1,0,0,0],
                   [0,1,0,-1],
                   [-1,0,1,0],
                   [0,0,0,1],
                   [1,1,0,0],
                   [0,0,1,1]], dtype=np.int64)
    Wb = np.array([[1,0,0,1],
                   [0,0,1,-1],
                   [0,1,0,1],
                   [1,0,1,0],
                   [-1,1,0,0],
                   [0,0,0,1],
                   [1,0,0,0]], dtype=np.int64)

    rnk = nb * 7  # 49

    # U: (n*m, rnk) = (4, 49)
    # U[i, 7*b + s] = Sr[i, s] for all b
    # This is Sr tiled nb times along axis 1
    U = np.tile(Sr, (1, nb))

    # V: (m*p, rnk) = (28, 49)
    # Row r = m*(2*b + lc) + lr  =>  lr + m*(2*b+lc)
    #   where lr in {0,1} (B row), b in 0..6 (block), lc in {0,1} (B col within block)
    # Col c = 7*b + s
    # V[r, c] = Vb[s, vc]  where vc = lr*2 + lc
    #
    # Build as: for each s (0..6), the column block (rows 0..27) is:
    #   for b in 0..6:
    #     for lr in 0,1:
    #       for lc in 0,1:
    #         V[lr + 2*(b*2+lc), 7*b+s] = Vb[s, lr*2+lc]
    #
    # Equivalent: V[:, 7*b+s] = vector of length 28:
    #   [Vb[s,0], Vb[s,1], Vb[s,2], Vb[s,3], Vb[s,0], Vb[s,1], Vb[s,2], Vb[s,3], ...] (repeated 7 times)
    # Wait, that's not right. Let me re-derive.
    #
    # Row index r: B is (m x p) = (2 x 14). Row r indexes B_flat[r] where
    #   B_flat[r] = B[r // p, r % p] = B[row, col] with row = r//14, col = r%14.
    # Actually for V: V has shape (m*p, rank) = (28, 49).
    # V[flat_row, col] contributes to the "V side" of tensor.
    # flat_row = b_row * p + b_col where b_row in 0..1, b_col in 0..13.
    # For block b, local col lc: b_col = 2*b + lc.
    # So flat_row = b_row * 14 + 2*b + lc.
    #
    # V[flat_row, 7*b + s] = Vb[s, vc] where vc maps (b_row, lc) -> index into (e,f,g,h):
    #   vc = b_row * 2 + lc  (0=e, 1=f, 2=g, 3=h)
    #
    # So for fixed s and b: the 4 rows (b_row=0,lc=0; b_row=0,lc=1; b_row=1,lc=0; b_row=1,lc=1)
    #   have flat_rows: 0*14+2b+0, 0*14+2b+1, 1*14+2b+0, 1*14+2b+1
    #   and values: Vb[s,0], Vb[s,1], Vb[s,2], Vb[s,3]
    #
    # Build V column by column (by s and b):
    V = np.zeros((m * p, rnk), dtype=np.int64)
    for s in range(7):
        for b in range(nb):
            col_idx = 7 * b + s
            V[0 * p + 2 * b + 0, col_idx] = Vb[s, 0]
            V[0 * p + 2 * b + 1, col_idx] = Vb[s, 1]
            V[1 * p + 2 * b + 0, col_idx] = Vb[s, 2]
            V[1 * p + 2 * b + 1, col_idx] = Vb[s, 3]

    # W: (p*n, rnk) = (28, 49)
    # W[flat_row, col] where flat_row = k*n + i, k=output col (0..13), i=output row (0..1)
    # For block b: output cols k in {2b, 2b+1}, output rows i in {0,1}
    # W[(2b+li)*2 + i, 7*b+s] = Wb[s, li*2 + i]  where li in {0,1} (col within block), i in {0,1} (row)
    # vc_w = li * 2 + i  (0=C11, 1=C12, 2=C21, 3=C22)
    W = np.zeros((p * n, rnk), dtype=np.int64)
    for s in range(7):
        for b in range(nb):
            col_idx = 7 * b + s
            W[(2*b + 0) * n + 0, col_idx] = Wb[s, 0]
            W[(2*b + 1) * n + 0, col_idx] = Wb[s, 1]
            W[(2*b + 0) * n + 1, col_idx] = Wb[s, 2]
            W[(2*b + 1) * n + 1, col_idx] = Wb[s, 3]

    if time.time() - _t0 > _DL:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
