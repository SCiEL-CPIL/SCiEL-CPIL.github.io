import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_factors_2x2x2():
    # Rank-7 Strassen factors for 2x2x2.
    # U[a*2+b, r], V[b*2+c, r], W[c*2+a, r]
    # r = 0: (a0+a1)(b0+b1)
    # r = 1: (a2+a3)b0
    # r = 2: a0(b2-b3)
    # r = 3: a3(b1-b0)
    # r = 4: (a0+a1)b3
    # r = 5: (a2-a0)(b0+b1)
    # r = 6: (a1-a3)(b2+b3)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1

    return U, V, W

def _build_4x4x4_strassen():
    # Build rank-49 factors for 4x4x4 by Kronecker product of 2x2x2 Strassen.
    U2, V2, W2 = _strassen_factors_2x2x2()

    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    # r = rb*7 + ri  (rb: block factor, ri: inner factor)
    for rb in range(7):
        for ri in range(7):
            r = rb * 7 + ri
            for ab in range(2):
                for bb in range(2):
                    ub = U2[ab * 2 + bb, rb]
                    for ai in range(2):
                        for bi in range(2):
                            uin = U2[ai * 2 + bi, ri]
                            val = ub * uin
                            if val == 0:
                                continue
                            a = ab * 2 + ai
                            b = bb * 2 + bi
                            U4[a * 4 + b, r] += val

            for bb in range(2):
                for cb in range(2):
                    vb = V2[bb * 2 + cb, rb]
                    for bi in range(2):
                        for ci in range(2):
                            vin = V2[bi * 2 + ci, ri]
                            val = vb * vin
                            if val == 0:
                                continue
                            b = bb * 2 + bi
                            c = cb * 2 + ci
                            V4[b * 4 + c, r] += val

            for cb in range(2):
                for ab in range(2):
                    wb = W2[cb * 2 + ab, rb]
                    for ci in range(2):
                        for ai in range(2):
                            win = W2[ci * 2 + ai, ri]
                            val = wb * win
                            if val == 0:
                                continue
                            c = cb * 2 + ci
                            a = ab * 2 + ai
                            W4[c * 4 + a, r] += val

    return U4, V4, W4

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # 4x4x8 = two independent 4x4x4 problems (B split into two 4-col halves).
    # Each uses a rank-49 Strassen decomposition. Total rank = 98.
    U4, V4, W4 = _build_4x4x4_strassen()

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # Left half (B cols 0..3 -> C cols 0..3)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4

    # Right half (B cols 4..7 -> C cols 4..7)
    U[:, 49:98] = U4
    # V rows for right half: b_global = b * 8 + (c + 4), with b in 0..3, c in 0..3
    for i in range(16):
        b = i // 4
        c = i % 4
        v_row = b * 8 + (c + 4)
        V[v_row, 49:98] = V4[i, :]
    # W rows for right half: c_global = (c + 4) * 4 + a
    for i in range(16):
        c = i // 4
        a = i % 4
        w_row = (c + 4) * 4 + a
        W[w_row, 49:98] = W4[i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("U shape:", len(res["U"]), "x", len(res["U"][0]))
    print("V shape:", len(res["V"]), "x", len(res["V"][0]))
    print("W shape:", len(res["W"]), "x", len(res["W"][0]))
    # Quick self-check
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    n, m, p = 4, 4, 8
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_ref = A @ B
    C_out = np.zeros((n, p))
    for r in range(len(U[0])):
        C_out += np.outer(W[:, r], U[:, r]) * V[:, r]
    print("Max abs error:", np.max(np.abs(C_out - C_ref)))
