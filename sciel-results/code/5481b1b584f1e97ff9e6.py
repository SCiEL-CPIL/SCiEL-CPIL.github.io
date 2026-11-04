import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START = time.time()


def _build_from_terms(terms):
    n, m, p = DIMENSIONS
    U = [[0.0] * len(terms) for _ in range(n * m)]
    V = [[0.0] * len(terms) for _ in range(m * p)]
    W = [[0.0] * len(terms) for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][c] = val
        for (j, k), val in v.items():
            V[j * p + k][c] = val
        for (i, k), val in w.items():
            W[k * n + i][c] = val
    return U, V, W


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    C = np.zeros((n, p))
    for c in range(Ua.shape[1]):
        Uc = Ua.reshape(n, m, -1)[:, :, c]
        Vc = Va.reshape(m, p, -1)[:, :, c]
        C += Uc @ Vc
    Wsum = np.zeros((n, p))
    for c in range(Wa.shape[1]):
        Wc = Wa.reshape(p, n, -1)[:, :, c]  # [k, i]
        Wsum += Wc.T                        # [i, k]
    if not np.allclose(C, Wsum, atol=1e-6):
        return False
    for arr in (Ua, Va, Wa):
        d = arr.ravel() * 2.0
        if np.any(np.abs(d - np.round(d)) > 1e-6):
            return False
    return True


def _schoolbook_terms():
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
    return terms


def _strassen_4x4_terms():
    terms = []
    A11 = (0, 0); A12 = (0, 2); A21 = (2, 0); A22 = (2, 2)
    B11 = (0, 0); B12 = (0, 2); B21 = (2, 0); B22 = (2, 2)
    C11 = (0, 0); C12 = (0, 2); C21 = (2, 0); C22 = (2, 2)

    def aidx(bl, r, inner):
        return (bl[0] + r, bl[1] + inner)

    def bidx(bl, inner, c):
        return (bl[0] + inner, bl[1] + c)

    def cidx(bl, r, c):
        return (bl[0] + r, bl[1] + c)

    Ms = [
        ([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)], [(C11, 1), (C22, 1)]),
        ([(A21, 1), (A22, 1)], [(B11, 1)], [(C21, 1), (C22, -1)]),
        ([(A11, 1)], [(B12, 1), (B22, -1)], [(C12, 1), (C22, 1)]),
        ([(A22, 1)], [(B21, 1), (B11, -1)], [(C11, 1), (C21, 1)]),
        ([(A11, 1), (A12, 1)], [(B22, 1)], [(C12, 1), (C11, -1)]),
        ([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)], [(C22, 1)]),
        ([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)], [(C11, 1)]),
    ]
    for u_blocks, v_blocks, targets in Ms:
        for r in range(2):
            for c in range(2):
                u_dict = {}
                for (ab, coeff) in u_blocks:
                    u_dict[aidx(ab, r, 0)] = u_dict.get(aidx(ab, r, 0), 0) + coeff
                    u_dict[aidx(ab, r, 1)] = u_dict.get(aidx(ab, r, 1), 0) + coeff
                v_dict = {}
                for (bb, coeff) in v_blocks:
                    v_dict[bidx(bb, 0, c)] = v_dict.get(bidx(bb, 0, c), 0) + coeff
                    v_dict[bidx(bb, 1, c)] = v_dict.get(bidx(bb, 1, c), 0) + coeff
                w_dict = {}
                for (cb, sign) in targets:
                    w_dict[cidx(cb, r, c)] = w_dict.get(cidx(cb, r, c), 0) + sign
                terms.append((u_dict, v_dict, w_dict))
    return terms


def _tail_schoolbook_terms():
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                terms.append({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
    return terms


def construct():
    best = None

    # Primary: Strassen on the 4x4x4 block + schoolbook on the 4x4x3 tail (rank 76)
    try:
        terms = _strassen_4x4_terms() + _tail_schoolbook_terms()
        U, V, W = _build_from_terms(terms)
        if _verify(U, V, W):
            best = (len(terms), U, V, W)
    except Exception:
        pass

    # Fallback: full schoolbook (always valid, rank 112)
    try:
        t = _schoolbook_terms()
        U, V, W = _build_from_terms(t)
        if _verify(U, V, W):
            if best is None or len(t) < best[0]:
                best = (len(t), U, V, W)
    except Exception:
        pass

    if best is None:
        t = _schoolbook_terms()
        U, V, W = _build_from_terms(t)
        return {"U": U, "V": V, "W": W}

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
