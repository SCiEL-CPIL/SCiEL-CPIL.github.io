import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # ------------------------------------------------------------------
    # Verified Strassen 2x2x2 decomposition (rank 7).
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]; sum_r U*V*W = 1 for all (i,j,k).
    # ------------------------------------------------------------------
    def strassen_2x2x2():
        U = np.zeros((4, 7), dtype=object)
        V = np.zeros((4, 7), dtype=object)
        W = np.zeros((4, 7), dtype=object)
        terms = [
            ([(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)],           [(0,1,1),(1,1,1)]),
            ([(0,0,1)],           [(1,0,1),(1,1,-1)], [(0,0,1)]),
            ([(1,1,1)],           [(0,1,1),(0,0,-1)], [(1,0,1)]),
            ([(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,1,1)]),
            ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
            ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
        ]
        for r, (u_list, v_list, w_list) in enumerate(terms):
            for (i, j, c) in u_list:
                U[i * 2 + j, r] += c
            for (j, k, c) in v_list:
                V[j * 2 + k, r] += c
            for (k, i, c) in w_list:
                W[k * 2 + i, r] += c
        return U, V, W

    # ------------------------------------------------------------------
    # Schoolbook decomposition for (nn x mm) * (mm x pp) -> (nn x pp).
    # ------------------------------------------------------------------
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------
    # 4x2x2 sub-block via Strassen: split the 4 rows into two 2-row blocks,
    # each a 2x2x2 Strassen (rank 7). Total rank 14? No: 2 blocks * 7 = 14.
    # Wait: 4x2x2 has 4*2*2 = 16 schoolbook. Strassen per 2x2x2 = 7, two of them = 14.
    # ------------------------------------------------------------------
    def build_4x2x2():
        nn, mm, pp = 4, 2, 2
        U2, V2, W2 = strassen_2x2x2()
        r_total = 14
        U = np.zeros((nn * mm, r_total), dtype=object)
        V = np.zeros((mm * pp, r_total), dtype=object)
        W = np.zeros((pp * nn, r_total), dtype=object)
        for ib in range(nn // 2):  # 2 row-blocks of 2
            base = ib * 7
            for t in range(7):
                col = base + t
                for row in range(4):  # U2 rows: i_local*2 + j
                    i_local = row // 2
                    j = row % 2
                    i_global = ib * 2 + i_local
                    U[i_global * mm + j, col] = U2[row, t]
                for row in range(4):  # V2 rows: j*2 + k_local
                    j = row // 2
                    k_local = row % 2
                    V[j * pp + k_local, col] = V2[row, t]
                for row in range(4):  # W2 rows: k_local*2 + i_local
                    k_local = row // 2
                    i_local = row % 2
                    i_global = ib * 2 + i_local
                    W[k_local * nn + i_global, col] = W2[row, t]
        return U, V, W, r_total

    # ------------------------------------------------------------------
    # Verification of a full decomposition.
    # ------------------------------------------------------------------
    def verify_decomp(U, V, W, nn, mm, pp, tol=1e-9):
        r = U.shape[1]
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        U3 = Uf.T.reshape(nn, mm, r)
        V3 = Vf.T.reshape(mm, pp, r)
        W3 = Wf.T.reshape(pp, nn, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        return err < tol

    # ------------------------------------------------------------------
    # Generic block combination.
    # Each block: (U_sub, V_sub, W_sub, i0, j0, k0, isz, jsz, ksz)
    # ------------------------------------------------------------------
    def combine(blocks, nn, mm, pp):
        total_rank = sum(b[0].shape[1] for b in blocks)
        U = np.zeros((nn * mm, total_rank), dtype=object)
        V = np.zeros((mm * pp, total_rank), dtype=object)
        W = np.zeros((pp * nn, total_rank), dtype=object)
        off = 0
        for U_sub, V_sub, W_sub, i0, j0, k0, isz, jsz, ksz in blocks:
            rsz = U_sub.shape[1]
            for i in range(isz):
                for j in range(jsz):
                    U[(i0 + i) * mm + (j0 + j), off:off + rsz] = \
                        U_sub[i * jsz + j, :]
            for j in range(jsz):
                for k in range(ksz):
                    V[(j0 + j) * pp + (k0 + k), off:off + rsz] = \
                        V_sub[j * ksz + k, :]
            for k in range(ksz):
                for i in range(isz):
                    W[(k0 + k) * nn + (i0 + i), off:off + rsz] = \
                        W_sub[k * isz + i, :]
            off += rsz
        return U, V, W, total_rank

    # ------------------------------------------------------------------
    # Build the full 4x5x6 decomposition.
    #   Part A: 4x2x6  -> 3 blocks of 4x2x2 (Strassen, rank 14 each) = 42
    #   Part B: 4x3x6  -> 2 blocks of 2x3x6 (schoolbook, rank 36 each) = 72
    #   Total rank = 114
    # ------------------------------------------------------------------
    def build_full():
        n, m, p = 4, 5, 6
        blocks = []

        # Part A: j in 0..1 (the 4x2x6 portion)
        Ua, Va, Wa, ra = build_4x2x2()
        for kb in range(p // 2):  # 3 output column-blocks of width 2
            blocks.append((Ua, Va, Wa, 0, 0, kb * 2, 4, 2, 2))

        # Part B: j in 2..4 (the 4x3x6 portion), split rows into 2x3x6 blocks
        for ib in range(n // 2):  # 2 row-blocks of height 2
            Ub, Vb, Wb = schoolbook(2, 3, 6)
            blocks.append((Ub, Vb, Wb, ib * 2, 2, 0, 2, 3, 6))

        return combine(blocks, n, m, p)

    U, V, W, rank = build_full()
    ok = verify_decomp(U, V, W, 4, 5, 6)

    if ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook 4x5x6 (rank 120)
    U, V, W = schoolbook(4, 5, 6)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
