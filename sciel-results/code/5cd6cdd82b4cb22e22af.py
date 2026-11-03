import numpy as np
import time

N, M, P = 2, 4, 5
NM, MP, PN = N * M, M * P, P * N

def construct():
    t0 = time.time()
    deadline = t0 + 100.0

    def build_schoolbook():
        r = N * M * P
        U = np.zeros((NM, r))
        V = np.zeros((MP, r))
        W = np.zeros((PN, r))
        idx = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, idx] = 1.0
                    V[j * P + k, idx] = 1.0
                    W[k * N + i, idx] = 1.0
                    idx += 1
        return U, V, W, r

    def validate(U, V, W):
        r = U.shape[1]
        result = np.einsum('ar,br,cr->abc', U, V, W)
        mask_ones = np.zeros((NM, MP, PN), dtype=bool)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    mask_ones[i * M + j, j * P + k, k * N + i] = True
        if not np.allclose(result[mask_ones], 1.0, atol=1e-8):
            return False
        mask_zeros = ~mask_ones
        if np.any(np.abs(result[mask_zeros]) > 1e-8):
            return False
        return True

    def to_output(U, V, W, r):
        def conv(mat):
            return [[float(mat[a, b]) for b in range(r)] for a in range(mat.shape[0])]
        return {"U": conv(U), "V": conv(V), "W": conv(W)}

    # Rank-36 construction
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    pat_221 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    R36 = 36
    U = np.zeros((NM, R36))
    V = np.zeros((MP, R36))
    W = np.zeros((PN, R36))
    col = 0
    for kb in range(2):
        for jb in range(2):
            for up, vp, wp in strassen_222:
                for ix, val in enumerate(up):
                    if val != 0:
                        il = ix // 2
                        jl = ix % 2
                        U[il * M + jb * 2 + jl, col] = val
                for ix, val in enumerate(vp):
                    if val != 0:
                        jl = ix // 2
                        kl = ix % 2
                        V[(jb * 2 + jl) * P + kb * 2 + kl, col] = val
                for ix, val in enumerate(wp):
                    if val != 0:
                        il = ix // 2
                        kl = ix % 2
                        W[(kb * 2 + kl) * N + il, col] = val
                col += 1
    for jb in range(2):
        for up, vp, wp in pat_221:
            for ix, val in enumerate(up):
                if val != 0:
                    il = ix // 2
                    jl = ix % 2
                    U[il * M + jb * 2 + jl, col] = val
            for ix, val in enumerate(vp):
                if val != 0:
                    jl = ix
                    V[(jb * 2 + jl) * P + 4, col] = val
            for ix, val in enumerate(wp):
                if val != 0:
                    il = ix
                    W[4 * N + il, col] = val
            col += 1

    assert col == R36
    if validate(U, V, W):
        return to_output(U, V, W, R36)

    Us, Vs, Ws, rs = build_schoolbook()
    return to_output(Us, Vs, Ws, rs)
