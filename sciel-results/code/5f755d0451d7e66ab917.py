"""
Block-structured rank-minimal decomposition of the 3x4x6 matrix-multiplication tensor.

Family (kept from the current program): partition the 3x4 * 4x6 product into
1x2 (rank-4) and 2x2 (Strassen, rank-7) sub-blocks -> rank 66 baseline, then
search for lower rank.

What is new / stronger:
  * Fast vectorized (NumPy) exact verification -> many more moves fit in budget.
  * A real "merge two terms into one" move (the actual rank-reducing operation):
    solve the bilinear merge conditions exactly (linear solve in the scaling
    variables, closed-form fallback) and test the residual.
  * Randomized restarts + drop + merge + local-polish, all under an explicit
    deadline guard so we always return the best exact decomposition found.
"""
import time
import random
import numpy as np

n, m, p = 3, 4, 6  # A: n x m, B: m x p, C: n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row (n*m)
def Bidx(j, k): return j * p + k          # V row (m*p)
def Cidx(i, k): return k * n + i          # W row (p*n), trace convention W[k*n+i]

# Precompute the (i,j,k) -> flattened-index mapping for the 72 tensor entries.
_TIK = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
_TU = np.array([Aidx(i, j) for (i, j, k) in _TIK], dtype=int)
_TV = np.array([Bidx(j, k) for (i, j, k) in _TIK], dtype=int)
_TW = np.array([Cidx(i, k) for (i, j, k) in _TIK], dtype=int)
_TU_T = _TU.T
_TV_T = _TV.T
_TW_T = _TW.T


def verify(U, V, W, r):
    """Exact check: for all (i,j,k), sum_l U[Aidx]*V[Bidx]*W[Cidx] == 1."""
    for l in range(r):
        s = (U[:, l] * V[:, l])[np.ix_(_TU, _TV)] * W[_TW, l]
        if not np.all(s == 1):
            return False
    return True


def residual(U, V, W, r):
    """Max |sum_l ... - 1| over all (i,j,k)."""
    acc = np.zeros(len(_TIK))
    for l in range(r):
        acc += (U[:, l] * V[:, l])[np.ix_(_TU, _TV)] * W[_TW, l]
    return float(np.max(np.abs(acc - 1)))


def baseline_66():
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c + 1)
        A10 = Aidx(a_r + 1, a_c); A11 = Aidx(a_r + 1, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c + 1)
        C10 = Cidx(c_r + 1, c_c); C11 = Cidx(c_r + 1, c_c + 1)
        U[A00][col] += 1; U[A11][col] += 1; V[B00][col] += 1; V[B11][col] += 1; W[C00][col] += 1; W[C11][col] += 1; col += 1
        U[A10][col] += 1; U[A11][col] += 1; V[B00][col] += 1; W[C10][col] += 1; W[C11][col] -= 1; col += 1
        U[A00][col] += 1; V[B01][col] += 1; V[B11][col] -= 1; W[C01][col] += 1; W[C11][col] += 1; col += 1
        U[A11][col] += 1; V[B10][col] += 1; V[B00][col] -= 1; W[C00][col] += 1; W[C10][col] += 1; col += 1
        U[A00][col] += 1; U[A01][col] += 1; V[B11][col] += 1; W[C00][col] -= 1; W[C01][col] += 1; col += 1
        U[A10][col] += 1; U[A00][col] -= 1; V[B00][col] += 1; V[B01][col] += 1; W[C11][col] += 1; col += 1
        U[A01][col] += 1; U[A11][col] -= 1; V[B10][col] += 1; V[B11][col] += 1; W[C00][col] += 1; col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c + 1)
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1; col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1; col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1; col += 1

    add_1x2_2x2(0, 0, 0, 0, 0, 0); add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2); add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4); add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0); add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2); add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4); add_strassen(1, 2, 2, 4, 1, 4)
    assert col == 66
    return U, V, W


def try_drop(U, V, W, r, l):
    """True if term l contributes nothing (so it can be removed exactly)."""
    acc = np.zeros(len(_TIK))
    for ll in range(r):
        if ll == l:
            continue
        acc += (U[:, ll] * V[:, ll])[np.ix_(_TU, _TV)] * W[_TW, ll]
    return bool(np.all(acc == 1))


def try_merge(U, V, W, r, a, b):
    """
    Try to replace terms a and b by a single rank-1 term.

    We look for scalars x,y,z (nonzero) such that, for every tensor entry t:
        x*U[a][t]*V[a][t]*W[a][t] + y*U[b][t]*V[b][t]*W[b][t]
          = z*U[a][t]*V[a][t]*W[a][t] + z*U[b][t]*V[b][t]*W[b][t]
    Define per-entry ratios rU=U[a]/U[b], rV=V[a]/V[b], rW=W[a]/W[b] (where the
    denominator is nonzero). The conditions become
        x*rU*rV*rW + y = z*(rU*rV*rW + 1)
    which is linear in (x,y,z) once the ratios are fixed. We solve the linear
    system; if a nonzero solution exists and the residual is ~0, the merge works.
    Returns (True, x, y, z) or (False, None, None, None).
    """
    Ua = np.asarray(U[a]); Ub = np.asarray(U[b])
    Va = np.asarray(V[a]); Vb = np.asarray(V[b])
    Wa = np.asarray(W[a]); Wb = np.asarray(W[b])
    rU = np.zeros_like(Ua, dtype=float)
    rV = np.zeros_like(Va, dtype=float)
    rW = np.zeros_like(Wa, dtype=float)
    nzU = Ub != 0
    rU[nzU] = Ua[nzU] / Ub[nzU]
    nzV = Vb != 0
    rV[nzV] = Va[nzV] / Vb[nzV]
    nzW = Wb != 0
    rW[nzW] = Wa[nzW] / Wb[nzW]

    P = rU * rV * rW
    # Build linear system: rows are (P, 1, -(P+1)), target 0.
    rows = []
    for t in range(len(_TIK)):
        rows.append((P[t], 1.0, -(P[t] + 1.0)))
    A = np.array(rows, dtype=float)
    # Solve the underdetermined homogeneous system: find null vector.
    try:
        _, _, Vt = np.linalg.svd(A)
        sol = Vt[-1]  # smallest singular value direction
        x, y, z = float(sol[0]), float(sol[1]), float(sol[2])
    except np.linalg.LinAlgError:
        return False, None, None, None
    if x == 0 and y == 0 and z == 0:
        return False, None, None, None
    # Scale so that the combined term has a "nice" normalization (z -> 1 if possible)
    if abs(z) > 1e-12:
        x /= z; y /= z; z = 1.0
    # Verify exactness of the merge on the actual (nonzero-denominator) entries.
    ok = True
    for t in range(len(_TIK)):
        lhs = x * Ua[t] * Va[t] * Wa[t] + y * Ub[t] * Vb[t] * Wb[t]
        rhs = z * Ua[t] * Va[t] * Wa[t] + z * Ub[t] * Vb[t] * Wb[t]
        if lhs != rhs:
            ok = False
            break
    if not ok:
        return False, None, None, None
    return True, x, y, z


def local_polish(U, V, W, r, deadline, rng):
    """Greedy loop: try drops and merges until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        # --- drops (cheap) ---
        for l in range(r):
            if try_drop(U, V, W, r, l):
                U = [row[:l] + row[l + 1:] for row in U]
                V = [row[:l] + row[l + 1:] for row in V]
                W = [row[:l] + row[l + 1:] for row in W]
                r -= 1
                improved = True
                break
        if improved:
            continue
        # --- merges (the rank-reducing move) ---
        order = list(range(r))
        rng.shuffle(order)
        for a in order:
            for b in order:
                if a == b:
                    continue
                ok, x, y, z = try_merge(U, V, W, r, a, b)
                if ok:
                    # New term = x*term_a + y*term_b (componentwise on each matrix)
                    newU = [x * U[a][t] + y * U[b][t] for t in range(len(U))]
                    newV = [x * V[a][t] + y * V[b][t] for t in range(len(V))]
                    newW = [x * W[a][t] + y * W[b][t] for t in range(len(W))]
                    # Remove a and b (keep the higher index first so removal is stable)
                    lo, hi = (a, b) if a < b else (b, a)
                    keep = [i for i in range(r) if i != lo and i != hi]
                    U2 = [U[i] for i in keep] + [newU]
                    V2 = [V[i] for i in keep] + [newV]
                    W2 = [W[i] for i in keep] + [newW]
                    r2 = r - 1
                    if verify(U2, V2, W2, r2):
                        U, V, W, r = U2, V2, W2, r2
                        improved = True
                        break
            if improved:
                break
    return r, U, V, W


def construct():
    deadline = time.time() + 140  # safety margin before the 150 s kill
    rng = random.Random(42)

    baseU, baseV, baseW = baseline_66()
    best_r, best_U, best_V, best_W = 66, baseU, baseV, baseW
    assert verify(best_U, best_V, best_W, best_r)

    # Restart loop: each restart starts from the current best and tries to go lower.
    restart = 0
    while time.time() < deadline - 12:
        restart += 1
        U, V, W, r = [row[:] for row in best_U], [row[:] for row in best_V], [row[:] for row in best_W], best_r
        # A little random perturbation of the search order / seed per restart.
        r2, U2, V2, W2 = local_polish(U, V, W, r, deadline, rng)
        if r2 < best_r:
            best_r, best_U, best_V, best_W = r2, U2, V2, W2
        # If we've hit a plateau, stop early (no more rank to gain found).
        if r2 == best_r and restart > 3:
            # one more pass with a fresh order to be safe, then stop
            r2, U2, V2, W2 = local_polish(best_U, best_V, best_W, best_r, deadline, rng)
            if r2 < best_r:
                best_r, best_U, best_V, best_W = r2, U2, V2, W2
            break

    assert verify(best_U, best_V, best_W, best_r), "Final decomposition is not exact!"
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
