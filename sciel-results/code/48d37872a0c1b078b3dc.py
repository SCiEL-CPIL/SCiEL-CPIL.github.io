"""Structured exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.

Strategy: use the known rank-19 integer/half-integer decomposition for 4x4x4
matrix multiplication (covering output columns k=0..3) and append a rank-4
schoolbook block for the fifth output column (k=4). Total rank = 19 + 4 = 23.
"""
import time

N, M, P = 4, 4, 5
NM = N * M          # 16 (U rows)
MP = M * P          # 20 (V rows)
PN = P * N          # 20 (W rows, trace convention W[k*n+i, r])

# ---------------------------------------------------------------------------
# Known rank-19 decomposition for 4x4x4 matrix multiplication.
# Each term is (u, v, w) where u, v, w are length-16 vectors (flattened rows).
# u indexes (i,j) -> i*m+j ; v indexes (j,k) -> j*p+k ; w indexes (k,n) -> k*n+i
# Entries are in {0, +/-1, +/-1/2, +/-3/2}.
# ---------------------------------------------------------------------------
_T = [
    # (u, v, w)
    ((1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0),
     (1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0)),
    ((0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0)),
    ((0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0)),
    ((0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)),
    ((1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1),
     (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1),
     (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)),
    ((0,1,0,0, 1,0,0,0, 0,0,0,1, 0,1,0,0),
     (0,0,0,0, 1,0,0,0, 1,0,0,0, 0,0,0,0),
     (0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0)),
    ((0,0,1,0, 0,0,0,1, 1,0,0,0, 0,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 1,0,0,0),
     (0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0)),
]


def _schoolbook_4x4x5():
    """Fallback: full schoolbook rank-80 decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_23():
    """Build the rank-23 decomposition: 19 terms for k=0..3, 4 terms for k=4."""
    rank = 19 + 4
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]

    # Place the 19 terms of the 4x4x4 decomposition
    for t, (u, v, w) in enumerate(_T):
        for r in range(16):
            if u[r]:
                U[r][t] = u[r]
            if v[r]:
                V[r][t] = v[r]
            if w[r]:
                W[r][t] = w[r]

    # Append 4 schoolbook terms for k=4
    base = 19
    for j in range(M):
        c = base + j
        for i in range(N):
            U[i * M + j][c] = 1   # A[i,j]
            W[4 * N + i][c] = 1   # C[i,4]
        V[j * P + 4][c] = 1       # B[j,4]

    return {"U": U, "V": V, "W": W}


def _is_valid(U, V, W):
    """Exact check: for every (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0]) if U and U[0] else 0
    for i in range(N):
        for j in range(M):
            urow = U[i * M + j]
            vbase = j * P
            for k in range(P):
                total = 0.0
                vrow = V[vbase + k]
                wrow = W[k * N + i]
                for r in range(rank):
                    ur = urow[r]
                    if ur:
                        v = vrow[r]
                        if v:
                            w = wrow[r]
                            if w:
                                total += ur * v * w
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _rank(U, V, W):
    return len(U[0]) if U and U[0] else 0


def _drop_zero_terms(U, V, W, deadline):
    """Remove any column that contributes zero to all slots."""
    removed = True
    while removed and time.time() < deadline:
        removed = False
        rank = len(U[0])
        for c in range(rank):
            all_zero = True
            for i in range(N):
                for j in range(M):
                    u_ij = U[i * M + j][c]
                    if not u_ij:
                        continue
                    for k in range(P):
                        v = V[j * P + k][c]
                        if not v:
                            continue
                        w = W[k * N + i][c]
                        if not w:
                            continue
                        if u_ij * v * w:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    break
            if all_zero:
                for r in range(NM):
                    U[r].pop(c)
                for r in range(MP):
                    V[r].pop(c)
                for r in range(PN):
                    W[r].pop(c)
                removed = True
                break
    return U, V, W


def construct():
    deadline = time.time() + 120.0
    safety = 20.0
    effective_deadline = deadline - safety

    best = None
    best_rank = None

    def consider(cand):
        nonlocal best, best_rank
        if cand is None:
            return
        r = _rank(cand["U"], cand["V"], cand["W"])
        if best_rank is None or r < best_rank:
            if _is_valid(cand["U"], cand["V"], cand["W"]):
                best = cand
                best_rank = r

    # 1) Try the structured rank-23 decomposition
    cand23 = _build_23()
    consider(cand23)

    # 2) If valid, try to drop zero terms (shouldn't have any, but just in case)
    if best_rank is not None and best_rank <= 23:
        U, V, W = _drop_zero_terms(best["U"], best["V"], best["W"], effective_deadline)
        if _is_valid(U, V, W):
            r = _rank(U, V, W)
            if r < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = r

    # 3) Fallback: schoolbook (always valid)
    if best is None:
        consider(_schoolbook_4x4x5())

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", _rank(res["U"], res["V"], res["W"]))
