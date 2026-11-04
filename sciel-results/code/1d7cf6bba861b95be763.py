import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    limit = t0 + 100.0

    # 4x4x7 tensor: C = A(4x4) @ B(4x7), W[k*n+i, c] for C[i,k]
    # Strategy: decompose into 4x4x4 (Strassen rank 49) + 4x4x3 (schoolbook rank 48)
    # Total rank = 97. But we can do better by using Strassen for the 4x4x3 part too.
    #
    # Better: split B into B1(4x4) and B2(4x3).
    # C1 = A @ B1 uses Strassen 4x4x4 (rank 49)
    # C2 = A @ B2: split into 3 columns, each 4x4x1
    # For 4x4x1, we can use rank-7 Strassen-like decomposition
    # Actually 4x4x1 = A @ B_col where B_col is 4x1
    # This is just a matrix-vector product, rank 16 (schoolbook) or can be reduced
    #
    # Let's use: 4x4x4 Strassen (49) + 4x4x2 Strassen-like (28) + 4x4x1 (16) = 93
    #
    # For 4x4x2: split into two 4x4x1, each can share structure
    # Actually, 4x4x2 via Strassen: treat as two columns
    #
    # Cleanest approach: build 4x4x4 Strassen, then add 3 schoolbook columns
    # Rank = 49 + 12 = 61... no, 4x4x1 schoolbook is 16, so 49+48=97
    #
    # Let's implement 4x4x4 Strassen properly, then 4x4x3 schoolbook

    # Standard Strassen 4x4x4: split into four 2x2 blocks
    # Use 7 recursive calls, each a 2x2x2 Strassen (rank 7)
    # Total: 7 * 7 = 49

    total_rank = 49 + 48  # 97
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0

    # ---- 4x4x4 Strassen decomposition ----
    # Split A into blocks: A00, A01, A10, A11 (each 2x2)
    # Split B1 (4x4) into blocks: B00, B01, B10, B11 (each 2x2)
    # 7 Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6

    # Each 2x2x2 Strassen has 7 rank-1 terms
    S2 = [
        # (x, y, c) for 2x2 Strassen, flat row-major 4-element
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [1,0,0,1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([1,0,-1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # Outer Strassen structure for 4x4x4
    # For each of 7 outer terms, specify which A-blocks, B-blocks, C-blocks are involved
    # Outer term t: A_blocks (block_idx -> coeff), B_blocks (block_idx -> coeff), C_blocks (pos -> coeff)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C positions: (row_block, col_block)

    outer = [
        # M1 = (A00+A11)*(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)*B00
        ({2:1, 3:1}, {0:1}, {(1,0):1, (1,1):-1}),
        # M3 = A00*(B01-B11)
        ({0:1}, {1:1, 3:-1}, {(0,1):1, (1,1):1}),
        # M4 = A11*(B10-B00)
        ({3:1}, {2:1, 0:-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)*B11
        ({0:1, 1:1}, {3:1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)*(B00+B01)
        ({2:1, 0:-1}, {0:1, 1:1}, {(1,1):1}),
        # M7 = (A01-A11)*(B10+B11)
        ({1:1, 3:-1}, {2:1, 3:1}, {(0,0):1}),
    ]

    for (ax, by, cc) in outer:
        for s2x, s2y, s2c in S2:
            c = col
            col += 1
            # Fill U: A contributions
            for bidx, a_coeff in ax.items():
                bi = bidx // 2  # row block
                bj = bidx % 2   # col block
                for li in range(2):
                    for lj in range(2):
                        s = s2x[li*2 + lj]
                        if s == 0:
                            continue
                        i = 2*bi + li
                        j = 2*bj + lj
                        U[i*m + j, c] += s * a_coeff
            # Fill V: B contributions (B is 4x4, cols 0-3)
            for bidx, b_coeff in by.items():
                bi = bidx // 2
                bj = bidx % 2
                for li in range(2):
                    for lj in range(2):
                        s = s2y[li*2 + lj]
                        if s == 0:
                            continue
                        i = 2*bi + li
                        j = 2*bj + lj
                        V[i*p + j, c] += s * b_coeff
            # Fill W: C contributions (C is 4x4, first 4 cols)
            for (rb, cb2), c_coeff in cc.items():
                for li in range(2):
                    for lj in range(2):
                        s = s2c[li*2 + lj]
                        if s == 0:
                            continue
                        i = 2*rb + li
                        k = 2*cb2 + lj
                        W[k*n + i, c] += s * c_coeff

    # ---- 4x4x3 schoolbook for B cols 4,5,6 ----
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                c = col
                col += 1
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0

    assert col == total_rank, f"rank mismatch: {col} != {total_rank}"

    # Verify
    Ua, Va, Wa = U, V, W
    ii = np.repeat(np.arange(n), m * p)
    jj = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n * m)
    ui = ii * m + jj
    vi = jj * p + kk
    wi = kk * n + ii
    check = (Ua[ui, :] * Va[vi, :] * Wa[wi, :]).sum(axis=1)
    if np.max(np.abs(check - 1.0)) > 1e-9:
        # Fallback: schoolbook rank 112
        rs = n * m * p
        Us = np.zeros((n * m, rs))
        Vs = np.zeros((m * p, rs))
        Ws = np.zeros((p * n, rs))
        cc2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, cc2] = 1
                    Vs[j * p + k, cc2] = 1
                    Ws[k * n + i, cc2] = 1
                    cc2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    if time.time() > limit:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
