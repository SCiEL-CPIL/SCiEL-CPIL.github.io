"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 2, 3]


# --- helpers -------------------------------------------------------------
def _build_matmul_tensor(n, m, p):
    """T[i*n+j, r] = prod_a M[i,a] * N[a,r] summed over a == 1.
    Index r uses the official trace convention W[k*n+i, r]."""
    T = np.zeros(n * p * n * m, dtype=float)
    for i in range(n):
        for j in range(m):
            for a in range(m):
                for r in range(p * n):
                    T[(i * m + j) * (p * n) + a * (p * n) + r] += (
                        1.0 if (i * m + a) == (r * m + j) else 0.0)
    return T


def _verify(n, m, p, U, V, W):
    """Return True iff U,V,W (list-of-lists) exactly decompose M_{n,m,p}
    under the convention T[i*m+j, r] = sum_c U[i*m+j,c]*V[j? ...].
    Convention used here (matching task contract):
        T[i*m+j, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    which equals the product coefficient M[i,a]N[a,r] with r=k*n+i.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    Ua3 = Ua.reshape(n, m, rank)
    Va3 = Va.reshape(m, p, rank)
    Wa3 = Wa.reshape(p, n, rank)
    ok = True
    for i in range(n):
        for j in range(m):
            for a in range(m):
                for r in range(p * n):
                    lhs = 1.0 if (i * m + a) == (r * m + j) else 0.0
                    acc = 0.0
                    for c in range(rank):
                        k, i2 = divmod(r, n)
                        acc += Ua3[i, j, c] * Va3[j, k, c] * Wa3[k, i2, c]
                    if abs(acc - lhs) > 1e-6:
                        ok = False
                        return False
    return ok


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _try_candidate(n, m, p, U, V, W):
    cand = {"U": U, "V": V, "W": W}
    try:
        if _verify(n, m, p, U, V, W):
            return cand
    except Exception:
        pass
    return None


# --- candidate 1: Strassen on first two columns + schoolbook on col 2 -----
def _strassen_plus_third(n, m, p):
    if m < 2 or p < 2:
        return None
    rank = 7 + (p - 2)  # 11 for p=3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen (2x2x2) on N columns 0,1 and full M; W index k*n+i with k in {0,1}
    # U[i*m+j] , V[j*p + k] (k in 0..1), W[k*n+i]
    strassen = [
        # (ui,j,i), (vj,k,j), (wk,i,i), coeff
        # s1 = (M00+M11)(N01+N10)  -> U[i,j] for M parts, V for N parts
        # We express each rank term U=alpha_e on row (i,j), V=beta on (j,k), W=gamma on (k,i)
        # Use the 7 products M_s * N_s:
        #   P1 = M(0,0+1,1)*N(0,0 +1,1)
        #   P2 = M(1,0)*      N(0,1 +1,1)
        #   P3 = M(0,0)*      N(1,0)
        #   P4 = M(1,1)*      N(1,1)
        #   P5 = M(0,0+1,0)*  N(1,1)
        #   P6 = M(1,0-1,1)*  N(0,0+1,1)
        #   P7 = M(0,1-1,1)*  N(0,0+1,0)
    ]
    # Encode as list of (Ucoeffs on (i,j) for i in 0..1,j in 0..1),
    #                     (Vcoeffs on (j,k) for j in 0..1, k in 0..1),
    #                     (Wcoeffs on (k,i) for k in 0..1, i in 0..1)
    def full(Um, Vn, Wm):
        # Um[(i,j)] is coefficient of M[i,j]; Vn[(j,k)] coeff of N[j,k]; Wm[(k,i)] coeff of output term (k,i)
        return Um, Vn, Wm

    terms = [
        # term: (dictU on (i,j), dictV on (j,k), dictW on (k,i))
        # P1
        ({(0,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
        # P2
        ({(1,0):1}, {(0,1):1,(1,1):1}, {(0,1):1,(1,1):1}),
        # P3
        ({(0,0):1}, {(1,0):1}, {(1,0):1}),
        # P4
        ({(1,1):1}, {(1,1):1}, {(1,1):1}),
        # P5
        ({(0,0):1,(1,0):1}, {(1,1):1}, {(0,1):1,(1,1):1}),
        # P6
        ({(1,0):1,(1,1):-1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(1,0):1}),
        # P7
        ({(0,1):1,(1,1):-1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1}),
    ]
    for idx, (dU, dV, dW) in enumerate(terms):
        for (i, j), cv in dU.items():
            U[i * m + j][idx] += cv
        for (j, k), cv in dV.items():
            V[j * p + k][idx] += cv
        for (k, i), cv in dW.items():
            W[k * n + i][idx] += cv

    # schoolbook for the remaining N columns (k = 2..p-1)
    col = 7
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    assert col == rank, (col, rank)
    return {"U": U, "V": V, "W": W}


def _empty(n, m, p):
    return None


# --- candidate 2/3: small coefficient families (generic rank-11 attempts) --
def _grid_candidate(n, m, p):
    # A parametric rank-11 family over first three N columns.
    # U depends only on (i,j); V on (j,k); W on (k,i) so the tensor factorizes
    # per (k) with shared U/V columns — a known necessary structure. Try integer
    # small coefficients; verification prunes the wrong ones.
    candidates = []
    # Strassen-3 style (7) plus one extra "mixed" rank term; enumerate a few
    # extra-term placements.
    base = _strassen_plus_third(n, m, p)
    if base is not None:
        candidates.append(base)
    # alternative: drop one schoolbook col-2 term and replace with a coupled term
    U0, V0, W0 = base["U"], base["V"], base["W"]
    for drop in range(0, 3):
        U = [row[:] for row in U0]
        V = [row[:] for row in V0]
        W = [row[:] for row in W0]
        rank = len(U[0])
        # remove a k=2 column term: col index for k=2 term is 7 + t where
        # t enumerates (i,j). For drop t:
        cidx = 7 + drop
        i, j = divmod(drop, m)
        U[i * m + j][cidx] = 0
        V[j * p + 2][cidx] = 0
        W[2 * n + i][cidx] = 0
        # add a new coupled term in the same column reusing Strassen U/V:
        U[i * m + j][cidx] = 1
        V[j * p + 2][cidx] = 1
        W[2 * n + i][cidx] = 1
        candidates.append({"U": U, "V": V, "W": W})
    return candidates


# --- main ----------------------------------------------------------------
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    built = _build_matmul_tensor(n, m, p)  # not strictly needed, kept for API parity

    candidates = []
    s1 = _strassen_plus_third(n, m, p)
    if s1 is not None:
        candidates.append(s1)
    for cand in _grid_candidate(n, m, p):
        if cand is not None:
            candidates.append(cand)

    for cand in candidates:
        ver = _try_candidate(n, m, p, cand["U"], cand["V"], cand["W"])
        if ver is not None:
            return ver

    # Fallback: guaranteed valid schoolbook (rank 12).
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
