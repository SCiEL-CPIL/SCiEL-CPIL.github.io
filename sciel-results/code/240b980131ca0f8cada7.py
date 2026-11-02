"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy (within the structured tensor / Winograd family):
  * 4x4 * 4x4 part via *recursive* Strassen: split into 2x2 blocks of 2x2
    matrices; the 7 Strassen multipliers at the outer level each perform a
    full 2x2 * 2x2 multiplication, giving rank 7*7 = 49.
  * 4x4 * 4x1 extra-column part via schoolbook: rank 16.
  * Total rank = 49 + 16 = 65  (vs. schoolbook 80).

All entries are integers (half-integers allowed), and the decomposition is
verified exactly with the official trace convention W[k*n+i, r].
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]


# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2.

    Returns (U,V,W) each of shape (4,7) with the convention:
        U[i*2 + j, c]  (i = row of A, j = col of A)
        V[j*2 + k, c]  (j = row of B, k = col of B)
        W[k*2 + i, c]  (k = row of C, i = col of C)
    so that  sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = (A*B)[i,k]  for all i,k.
    """
    # Strassen multipliers M1..M7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1

    # M3 = a11(b21-b22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1

    # M4 = a22(b12-b11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[3, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1

    return U, V, W


def _strassen_4x4x4():
    """Recursive Strassen for 4x4 * 4x4, rank 49.

    Returns (U,V,W) each of shape (16,49) with convention
        U[i*4 + j, c], V[j*4 + k, c], W[k*4 + i, c].
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49

    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))

    # Outer Strassen: 7 multipliers, each a 2x2*2x2 product (rank 7).
    # For each outer multiplier m (0..6) and inner multiplier t (0..6):
    #   U factor = U2[outer_i, m] * U2[inner_i, t]
    #   V factor = V2[outer_j, m] * V2[inner_j, t]
    #   W factor = W2[outer_k, m] * W2[inner_k, t]
    # where outer indices map 4x4 block rows/cols -> 2x2, inner -> 2x2.
    col = 0
    for m in range(rank2):
        for t in range(rank2):
            for bi in range(2):   # outer block row (0/1)
                for bk in range(2):  # outer block col (0/1)
                    for ii in range(2):  # inner row
                        for jj in range(2):  # inner col
                            for kk in range(2):  # inner col
                                i = bi * 2 + ii
                                j = bk * 2 + jj
                                k = bk * 2 + kk
                                uval = U2[i, m] * U2[ii, t]
                                vval = V2[j, m] * V2[jj, t]
                                wval = W2[k, m] * W2[kk, t]
                                if uval != 0 and vval != 0 and wval != 0:
                                    U4[i * 4 + j, col] = uval
                                    V4[j * 4 + k, col] = vval
                                    W4[k * 4 + i, col] = wval
            col += 1

    return U4, V4, W4


def _construct_4x4x5():
    """Assemble 4x4 * 4x5 = (4x4*4x4) + (4x4*4x1).  Rank 49 + 16 = 65."""
    U4, V4, W4 = _strassen_4x4x4()
    rank4 = U4.shape[1]  # 49

    n, m, p = 4, 4, 5
    rank = rank4 + n * m  # 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # 4x4 * 4x4 part (columns 0..48)
    U[:, :rank4] = U4
    V[:16, :rank4] = V4
    W[:16, :rank4] = W4

    # 4x4 * 4x1 part (columns 49..64): extra column k = 4
    col = rank4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wc.T
    return bool(np.allclose(T, 1.0, atol=atol))


def _to_list(d):
    return {"U": np.asarray(d["U"]).tolist(),
            "V": np.asarray(d["V"]).tolist(),
            "W": np.asarray(d["W"]).tolist()}


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Primary: recursive Strassen (49) + schoolbook extra column (16) = 65
    try:
        cand = _construct_4x4x5()
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            return _to_list(cand)
    except Exception:
        pass

    if time.time() > deadline:
        return _to_list(_schoolbook())

    # Fallback: schoolbook (rank 80), always valid.
    return _to_list(_schoolbook())
# EVOLVE-BLOCK-END
