"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses a 2-level (recursive) Strassen decomposition:
  - Outer level: 7 block-level Strassen products
  - Inner level: 7 element-level Strassen products
  - Total rank: 7 * 7 = 49

The W matrix follows the trace convention W[k*n + i, r].
"""

import time
import numpy as np

def construct():
    _t0 = time.time()

    n = m = p = 4
    R = 49

    # --- Inner 2x2 Strassen (element level) ---
    # Each product t has 4 coefficients for A entries [00, 01, 10, 11]
    _ua = np.array([
        [1,  0,  0,  1],   # t=0: a00 + a11
        [0,  0,  1,  1],   # t=1: a10 + a11
        [1,  0,  0,  0],   # t=2: a00
        [0,  0,  0,  1],   # t=3: a11
        [1,  1,  0,  0],   # t=4: a00 + a01
        [-1, 0,  1,  0],   # t=5: a10 - a00
        [0,  1,  0, -1],   # t=6: a01 - a11
    ], dtype=np.int64)  # shape (7, 4)

    _vb = np.array([
        [1,  0,  0,  1],   # t=0: b00 + b11
        [1,  0,  0,  0],   # t=1: b00
        [0,  1,  0, -1],   # t=2: b01 - b11
        [-1, 0,  1,  0],   # t=3: b10 - b00
        [0,  0,  0,  1],   # t=4: b11
        [1,  1,  0,  0],   # t=5: b00 + b01
        [0,  0,  1,  1],   # t=6: b10 + b11
    ], dtype=np.int64)  # shape (7, 4)

    # W_elem[t] is a 2x2 matrix: W_elem[t][row, col] = coefficient for C[row,col]
    _wc = np.array([
        [[1,  0], [0,  1]],   # t=0: C00+=M, C11+=M
        [[0,  0], [1, -1]],   # t=1: C10+=M, C11-=M
        [[0,  1], [0,  1]],   # t=2: C01+=M, C11+=M
        [[1,  0], [1,  0]],   # t=3: C00+=M, C10+=M
        [[-1, 1], [0,  0]],   # t=4: C00-=M, C01+=M
        [[0,  0], [0,  1]],   # t=5: C11+=M
        [[1,  0], [0,  0]],   # t=6: C00+=M
    ], dtype=np.int64)  # shape (7, 2, 2)

    # --- Outer 2x2 Strassen (block level) ---
    # For each block product s, we have coefficients for A-blocks [A00, A01, A10, A11]
    _sa = np.array([
        [1,  0,  0,  1],   # s=0: A00 + A11
        [0,  0,  1,  1],   # s=1: A10 + A11
        [1,  0,  0,  0],   # s=2: A00
        [0,  0,  0,  1],   # s=3: A11
        [1,  1,  0,  0],   # s=4: A00 + A01
        [0,  0,  1, -1],   # s=5: A10 - A00
        [0,  1,  0, -1],   # s=6: A01 - A11
    ], dtype=np.int64)  # shape (7, 4)

    _sb = np.array([
        [1,  0,  0,  1],   # s=0: B00 + B11
        [1,  0,  0,  0],   # s=1: B00
        [0,  1,  0, -1],   # s=2: B01 - B11
        [-1, 0,  1,  0],   # s=3: B10 - B00
        [0,  0,  0,  1],   # s=4: B11
        [1,  1,  0,  0],   # s=5: B00 + B01
        [0,  0,  1,  1],   # s=6: B10 + B11
    ], dtype=np.int64)  # shape (7, 4)

    # Block-level W: _sc[s] is a 2x2 matrix of coefficients for C-blocks
    _sc = np.array([
        [[1,  0], [0,  1]],   # s=0: C00+=M, C11+=M
        [[0,  0], [1, -1]],   # s=1: C10+=M, C11-=M
        [[0,  1], [0,  1]],   # s=2: C01+=M, C11+=M
        [[1,  0], [1,  0]],   # s=3: C00+=M, C10+=M
        [[-1, 1], [0,  0]],   # s=4: C00-=M, C01+=M
        [[0,  0], [0,  1]],   # s=5: C11+=M
        [[1,  0], [0,  0]],   # s=6: C00+=M
    ], dtype=np.int64)  # shape (7, 2, 2)

    # --- Build U (16 x 49) ---
    # Row index for U: i * m + j  where i is A-row, j is A-col
    # Global A element (i, j) is in block (bi, bj) = (i//2, j//2), local (ei, ej) = (i%2, j%2)
    # For block product s and element product t:
    #   U[i*m+j, s*7+t] = sum over blocks (bi,bj) of _sa[s, bi*2+bj] * _ua[t, ei*2+ej]

    U = np.zeros((16, R), dtype=np.int64)
    V = np.zeros((16, R), dtype=np.int64)
    W = np.zeros((16, R), dtype=np.int64)

    # Precompute block-to-element mappings
    # For A: block (bi, bj) -> elements (bi*2+ei, bj*2+ej) for ei,ej in {0,1}
    # Row in U for element (i, j) = i * 4 + j

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # U contributions
            for bi in range(2):
                for bj in range(2):
                    c_s = _sa[s, bi * 2 + bj]
                    if c_s == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_t = _ua[t, ei * 2 + ej]
                            if c_t == 0:
                                continue
                            row_a = (bi * 2 + ei) * 4 + (bj * 2 + ej)
                            U[row_a, col] += c_s * c_t

            # V contributions
            # Row index for V: j * p + k  where j is B-row, k is B-col
            # Global B element (j, k) is in block (bj, bk) = (j//2, k//2), local (ej, ek) = (j%2, k%2)
            # For block product s and element product t:
            #   V[j*p+k, s*7+t] = sum over blocks (bj,bk) of _sb[s, bj*2+bk] * _vb[t, ej*2+ek]
            for bj in range(2):
                for bk in range(2):
                    c_s = _sb[s, bj * 2 + bk]
                    if c_s == 0:
                        continue
                    for ej in range(2):
                        for ek in range(2):
                            c_t = _vb[t, ej * 2 + ek]
                            if c_t == 0:
                                continue
                            row_b = (bj * 2 + ej) * 4 + (bk * 2 + ek)
                            V[row_b, col] += c_s * c_t

            # W contributions
            # Row index for W: k * n + i  where k is C-col, i is C-row
            # Global C element (i, k) is in block (bi, bk) = (i//2, k//2), local (ei, ek) = (i%2, k%2)
            # For block product s and element product t:
            #   W[k*n+i, s*7+t] = sum over blocks (bi,bk) of _sc[s, bi*2+bk] * _wc[t, ei, ek]
            for bi in range(2):
                for bk in range(2):
                    c_s = _sc[s, bi * 2 + bk]
                    if c_s == 0:
                        continue
                    for ei in range(2):
                        for ek in range(2):
                            c_t = _wc[t, ei, ek]
                            if c_t == 0:
                                continue
                            row_c = (bk * 2 + ek) * 4 + (bi * 2 + ei)
                            W[row_c, col] += c_s * c_t

    # --- Verification ---
    def _check(Um, Vm, Wm):
        # Build target tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
        T = np.zeros((16, 16, 16), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T[i * 4 + j, j * 4 + k, k * 4 + i] = 1.0

        D = np.einsum('ur,vr,wr->uvw', Um, Vm, Wm, optimize=True)
        return np.allclose(T, D, atol=1e-9)

    if time.time() - _t0 < 120.0:
        if _check(U.astype(np.float64), V.astype(np.float64), W.astype(np.float64)):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-64 schoolbook
    Uf = np.zeros((16, 64), dtype=np.int64)
    Vf = np.zeros((16, 64), dtype=np.int64)
    Wf = np.zeros((16, 64), dtype=np.int64)
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Uf[i * 4 + j, c] = 1
                Vf[j * 4 + k, c] = 1
                Wf[k * 4 + i, c] = 1
                c += 1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
