import numpy as np
import time

def construct():
    t0 = time.time()
    tlimit = t0 + 115

    nn, mm, pp = 4, 5, 6
    NU = nn * mm          # 20
    NV = mm * pp          # 30
    NW = pp * nn          # 24

    # Strassen 2x2x2: 7 rank-1 terms
    STR = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]

    RANK = 96
    U = np.zeros((NU, RANK), dtype=np.float64)
    V = np.zeros((NV, RANK), dtype=np.float64)
    W = np.zeros((NW, RANK), dtype=np.float64)

    col = 0

    def add_block_2x2x2(i0, j0, k0):
        nonlocal col
        for (u_mat, v_mat, w_mat) in STR:
            for li in range(2):
                for lj in range(2):
                    c = u_mat[li][lj]
                    if c != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * mm + gj, col] += c
            for lj in range(2):
                for lk in range(2):
                    c = v_mat[lj][lk]
                    if c != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * pp + gk, col] += c
            for lk in range(2):
                for li in range(2):
                    c = w_mat[lk][li]
                    if c != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * nn + gi, col] += c
            col += 1

    def add_block_2x1x2(i0, j0, k0):
        nonlocal col
        for li in range(2):
            for lk in range(2):
                gi = i0 + li
                gj = j0
                gk = k0 + lk
                U[gi * mm + gj, col] = 1.0
                V[gj * pp + gk, col] = 1.0
                W[gk * nn + gi, col] = 1.0
                col += 1

    def add_block_2x3x2(i0, j0, k0):
        add_block_2x2x2(i0, j0, k0)
        add_block_2x1x2(i0, j0 + 2, k0)

    def add_block_2x5x2(i0, j0, k0):
        add_block_2x2x2(i0, j0, k0)
        add_block_2x3x2(i0, j0 + 2, k0)

    def add_block_2x5x6(i0, j0, k0):
        add_block_2x5x2(i0, j0, k0)
        add_block_2x5x2(i0, j0, k0 + 2)
        add_block_2x5x2(i0, j0, k0 + 4)

    # Decompose 4x5x6 into two 2x5x6 blocks
    add_block_2x5x6(0, 0, 0)
    add_block_2x5x6(2, 0, 0)

    if time.time() > tlimit:
        return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}

    assert col == RANK, f"col={col}, expected {RANK}"

    # Verify
    Ua = U[:, :col]
    Va = V[:, :col]
    Wa = W[:, :col]

    max_err = 0.0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                ui = Ua[i * mm + j, :]
                vj = Va[j * pp + k, :]
                wk = Wa[k * nn + i, :]
                s = np.dot(ui, vj * wk)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback: pure schoolbook
        rs = nn * mm * pp
        Us = np.zeros((NU, rs), dtype=np.float64)
        Vs = np.zeros((NV, rs), dtype=np.float64)
        Ws = np.zeros((NW, rs), dtype=np.float64)
        c2 = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    Us[i * mm + j, c2] = 1.0
                    Vs[j * pp + k, c2] = 1.0
                    Ws[k * nn + i, c2] = 1.0
                    c2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
