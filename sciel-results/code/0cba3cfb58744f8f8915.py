"""Matrix multiplication 4x4x4 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    """
    start = time.time()
    limit = start + 140

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 factor matrices (4,7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    # M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    # M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    # M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    r = 49
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)

    # Fill via explicit Kronecker expansion
    for ii in range(n):
        a = ii >> 1
        b = ii & 1
        for jj in range(m):
            c = jj >> 1
            d = jj & 1
            row = ii * m + jj
            uA = U2[a * 2 + c]
            uB = U2[b * 2 + d]
            for cc in range(7):
                v1 = uA[cc]
                if v1 == 0: continue
                base = cc * 7
                for dd in range(7):
                    val = v1 * uB[dd]
                    if val != 0:
                        U[row, base + dd] = val

    for jj in range(m):
        c = jj >> 1
        d = jj & 1
        for kk in range(p):
            e = kk >> 1
            f = kk & 1
            row = jj * p + kk
            vA = V2[c * 2 + e]
            vB = V2[d * 2 + f]
            for cc in range(7):
                v1 = vA[cc]
                if v1 == 0: continue
                base = cc * 7
                for dd in range(7):
                    val = v1 * vB[dd]
                    if val != 0:
                        V[row, base + dd] = val

    for kk in range(p):
        e = kk >> 1
        f = kk & 1
        for ii in range(n):
            a = ii >> 1
            b = ii & 1
            row = kk * n + ii
            wA = W2[e * 2 + a]
            wB = W2[f * 2 + b]
            for cc in range(7):
                v1 = wA[cc]
                if v1 == 0: continue
                base = cc * 7
                for dd in range(7):
                    val = v1 * wB[dd]
                    if val != 0:
                        W[row, base + dd] = val

    if time.time() > limit:
        raise TimeoutError()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
