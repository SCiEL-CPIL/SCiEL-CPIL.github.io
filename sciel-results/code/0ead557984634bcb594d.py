import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Strassen 2x2x2 definitions
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U factors (A), V factors (B), W factors (C)
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00+a11
        {2: 1, 3: 1},       # P2: a10+a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00+a01
        {2: 1, 0: -1},      # P6: a10-a00
        {1: 1, 3: -1},      # P7: a01-a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b00+b11
        {0: 1},             # P2: b00
        {1: 1, 3: -1},      # P3: b01-b11
        {2: 1, 0: -1},      # P4: b10-b00
        {3: 1},             # P5: b11
        {0: 1, 1: 1},       # P6: b00+b01
        {2: 1, 3: 1},       # P7: b10+b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00+c11
        {2: 1, 3: -1},      # P2: c10-c11
        {1: 1, 3: 1},       # P3: c01+c11
        {0: 1, 2: 1},       # P4: c00+c10
        {0: -1, 1: 1},      # P5: -c00+c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]

    # Total rank: 6 blocks * 7 terms = 42
    rank_total = 42
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))

    col = 0
    # Blocks:
    # jm: row block of B (0 or 1). Corresponds to rows 2*jm, 2*jm+1 of B.
    #      Also corresponds to col block of A? 
    #      A is 2x4. B is 4x6.
    #      C = A @ B.
    #      C[i, k] = sum_j A[i, j] B[j, k].
    #      We split j into 2 blocks of 2.
    #      Block jm uses A[:, 2*jm : 2*jm+2] and B[2*jm : 2*jm+2, :].
    #      We split k into 3 blocks of 2.
    #      Block kp uses B[:, 2*kp : 2*kp+2] and C[:, 2*kp : 2*kp+2].
    
    for jm in range(2):      # A col block / B row block
        for kp in range(3):  # B col block / C col block
            for pi in range(7):
                r = col + pi
                # U: A[i, 2*jm + j_local]
                # i in 0..1, j_local in 0..1
                for loc, coeff in strassen_U[pi].items():
                    li, lj = divmod(loc, 2)
                    # U index: i * m + j_global
                    # j_global = 2*jm + lj
                    U[li * m + (2 * jm + lj), r] = coeff
                
                # V: B[2*jm + j_local, 2*kp + k_local]
                # j_global = 2*jm + lj, k_global = 2*kp + lk
                for loc, coeff in strassen_V[pi].items():
                    lj, lk = divmod(loc, 2)
                    # V index: j * p + k
                    V[(2 * jm + lj) * p + (2 * kp + lk), r] = coeff
                
                # W: C[2*kp + k_local, i]
                # k_global = 2*kp + lk, i in 0..1
                for loc, coeff in strassen_W[pi].items():
                    li, lk = divmod(loc, 2)
                    # W index: k * n + i
                    W[(2 * kp + lk) * n + li, r] = coeff
            col += 7

    # Verification (fast)
    # Check a few random entries or full tensor if fast enough
    # Full tensor check is 2*4*6*42 = 2016 multiplications. Very fast.
    T_expected = np.ones((n, m, p))
    T_actual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank_total):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                T_actual[i, j, k] = s
    
    if not np.allclose(T_actual, T_expected, atol=1e-10):
        # Fallback to schoolbook (rank 48) just in case, though unlikely
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, r] = 1
                    V_sb[j * p + k, r] = 1
                    W_sb[k * n + i, r] = 1
                    r += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
