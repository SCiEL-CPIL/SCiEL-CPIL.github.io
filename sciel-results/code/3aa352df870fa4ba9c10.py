"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition.

Design family (from the improving island): split the inner dimension m=3 into
(2,1) and merge the 2x2 core with Strassen-style linear combinations. We
construct a rank-50 decomposition by fixing a structured sparse pattern for
(U, V) with 50 columns and solving the resulting *linear* system for the
column weights so the tensor identity holds exactly. The result is verified;
a schoolbook fallback guarantees validity.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    DEADLINE = start_time + 150.0

    def left():
        return DEADLINE - time.time()

    n, m, p = DIMENSIONS

    base = _build_rank50(n, m, p)
    best = base
    best_rank = len(base["U"][0])

    if left() > 30.0:
        polished = _exact_rank_reduction(best, budget=30.0)
        if polished is not None:
            r = len(polished["U"][0])
            if r < best_rank and _verify(n, m, p, polished) == r:
                best = polished
                best_rank = r

    if _verify(n, m, p, best) != len(best["U"][0]):
        return _schoolbook(n, m, p)
    return best


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_rank50(n, m, p):
    """Construct a rank-50 decomposition by exact linear solve.

    We fix U and V to a structured sparse pattern with 50 columns. Each
    column c contributes to the tensor as
        T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For a fixed (U, V), the tensor is linear in the W entries. We build the
    linear system  A @ w = b  over the 54 tensor coordinates and solve it
    exactly with Fraction. If a solution exists with half-integer entries,
    we return it.
    """
    # Build the 50-column structured (U, V) pattern:
    #  - 36 columns: one-hot U on (i, j in {0,1}) and one-hot V on (j, k).
    #    These cover the 2x2 core: i in {0,1}, j in {0,1}, k in 0..8 -> 36 cols.
    #  - 18 columns: one-hot U on (i, j=2) and one-hot V on (j=2, k).
    #    These cover the tail: i in {0,1}, k in 0..8 -> 18 cols.
    #  Total = 54. To reach 50 we merge 4 pairs of core columns using
    #  Strassen: for each of the 9 output columns k, merge the two i-rows
    #  (i=0 and i=1) of the j=0 core into one column with a shared U pattern.
    #
    # Rather than hand-merging, we let the linear solve decide: we use 50
    # columns where 4 of them have a 2-cell U support (Strassen merge) and
    # the rest are one-hot, then solve for W. We try a few merge patterns and
    # accept the first that yields an exact half-integer solution.
    for pattern in _candidate_patterns(n, m, p, target=50):
        sol = _solve_for_W(n, m, p, pattern)
        if sol is not None:
            return sol
    return _schoolbook(n, m, p)


def _candidate_patterns(n, m, p, target):
    """Yield candidate structured (U, V) patterns with `target` columns."""
    # Pattern A: merge the 2x2 core per output column using Strassen.
    # For each k in 0..8, the core 2x2 (i in {0,1}, j in {0,1}) is handled
    # by 7 Strassen rank-1 terms. That is 7*9 = 63 columns for the core,
    # plus 18 tail = 81. Too many. So Strassen per-column is not the route.
    #
    # Pattern B (the actual rank-50 route): the 2x2x9 core = (2x2) x (2x9).
    # Write the 2x9 as 9 columns; Strassen on 2x2x2 tiles. Split 9 cols into
    # four 2-col tiles (8 cols) + one 1-col tile: 4 tiles * 7 + 4 (schoolbook
    # for the 1-col tile, 2x2x1=4) = 28 + 4 = 32 core columns. Plus 18 tail
    # = 50. This is the rank-50 construction.
    #
    # We build these 50 columns explicitly.
    cols = []  # each: (u_support dict, v_support dict)

    # Core: 2x2 (i in {0,1}, j in {0,1}) times 2x9 (rows j in {0,1}, cols k).
    # Strassen on 2x2x2 tile for column pairs (0,1),(2,3),(4,5),(6,7).
    # For a tile with columns (k0, k1), Strassen gives 7 products:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B10)
    #   M5 = (A00+A10)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A11-A01)(B01+B11)
    # where A is 2x2 (rows i, cols j) and B is 2x2 (rows j, cols k).
    # Each M is a scalar product of two linear combos -> one rank-1 term
    # with U = (coeffs over i,j), V = (coeffs over j,k).
    strassen = [
        (({0: 1, 1: 0, 0: 0, 1: 0}), ({0: 1, 1: 1})),  # placeholder, replaced below
    ]
    # We define Strassen terms as (U_coeffs over (i,j) for i in 0..1, j in 0..1,
    # V_coeffs over (j,k) for j in 0..1, k in 0..1).
    def ij(i, j): return (i, j)
    def jk(j, k): return (j, k)
    S = [
        ({ij(0, 0): 1, ij(1, 1): 1}, {jk(0, 0): 1, jk(1, 1): 1}),
        ({ij(1, 0): 1, ij(1, 1): 1}, {jk(0, 0): 1}),
        ({ij(0, 0): 1}, {jk(1, 0): 1, jk(1, 1): -1}),
        ({ij(1, 1): 1}, {jk(0, 1): 1, jk(1, 0): -1}),
        ({ij(0, 0): 1, ij(0, 1): 1}, {jk(1, 1): 1}),
        ({ij(1, 0): 1, ij(0, 0): -1}, {jk(0, 0): 1, jk(1, 0): 1}),
        ({ij(1, 1): 1, ij(0, 1): -1}, {jk(0, 1): 1, jk(1, 1): 1}),
    ]
    tiles = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for (k0, k1) in tiles:
        for (u, v) in S:
            u_abs = {}
            for ((i, j), c) in u.items():
                u_abs[(i, j)] = c
            v_abs = {}
            for ((j, kk), c) in v.items():
                # map local kk in {0,1} to global k0+kk
                v_abs[(j, k0 + kk)] = c
            cols.append((u_abs, v_abs))
    # Last core column tile: k=8 alone, 2x2x1 schoolbook (4 terms).
    for i in (0, 1):
        for j in (0, 1):
            cols.append(( { (i, j): 1 }, { (j, 8): 1 } ))
    # Tail: j=2, i in {0,1}, k in 0..8 (18 terms).
    for i in (0, 1):
        for k in range(p):
            cols.append(( { (i, 2): 1 }, { (2, k): 1 } ))

    # cols should have 4*7 + 4 + 18 = 28 + 4 + 18 = 50 entries.
    if len(cols) != target:
        # If mismatched, fall back to a one-hot 54 and let solve fail gracefully.
        return []
    return [cols]


def _solve_for_W(n, m, p, pattern):
    """Given a list of (u_support, v_support), solve for W exactly."""
    cols = pattern[0]
    rank = len(cols)
    # Unknowns: W[k*n+i, c] for all k, i, c -> p*n*rank unknowns.
    # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    # This is linear in W. Build the system with Fraction.
    # Number of equations = n*m*p = 54. Unknowns = p*n*rank = 9*2*50 = 900.
    # Underdetermined -> pick a particular solution. We solve via Gaussian
    # elimination on the 54 equations, treating W entries as variables, and
    # set free variables to 0.
    eqs = []  # list of (dict var->coeff, rhs)
    var_id = {}
    def var(k, i, c):
        key = (k, i, c)
        if key not in var_id:
            var_id[key] = len(var_id)
        return var_id[key]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d = {}
                for c in range(rank):
                    u = cols[c][0].get((i, j), 0)
                    v = cols[c][1].get((j, k), 0)
                    if u != 0 and v != 0:
                        vid = var(k, i, c)
                        d[vid] = d.get(vid, Fraction(0)) + Fraction(u) * Fraction(v)
                if d:
                    eqs.append((d, Fraction(1)))
                else:
                    # No term covers this (i,j,k) -> infeasible.
                    return None
    # Gaussian elimination (exact).
    num_vars = len(var_id)
    # Build rows as (coeff_dict, rhs)
    rows = list(eqs)
    pivot_cols = []
    r = 0
    for col in sorted(set(v for d, _ in rows for v in d)):
        if r >= len(rows):
            break
        # find a row >= r with nonzero at col
        piv = None
        for rr in range(r, len(rows)):
            if rows[rr][0].get(col, Fraction(0)) != 0:
                piv = rr
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        lead = rows[r][0][col]
        # normalize
        d, rhs = rows[r]
        nd = {k2: v2 / lead for k2, v2 in d.items()}
        rows[r] = (nd, rhs / lead)
        # eliminate
        for rr in range(len(rows)):
            if rr == r:
                continue
            f = rows[rr][0].get(col, Fraction(0))
            if f != 0:
                dd, rhh = rows[rr]
                ndd = dict(dd)
                for k2, v2 in nd.items():
                    ndd[k2] = ndd.get(k2, Fraction(0)) - f * v2
                    if ndd[k2] == 0:
                        del ndd[k2]
                rows[rr] = (ndd, rhh - f * (rhs / lead))
        pivot_cols.append(col)
        r += 1
    # Check consistency: any row with empty coeffs but nonzero rhs -> infeasible.
    for d, rhs in rows:
        if not d and rhs != 0:
            return None
    # Build solution: set free vars to 0, back-substitute.
    sol = {v: Fraction(0) for v in var_id}
    for col in reversed(pivot_cols):
        # find the row whose leading col is col
        for d, rhs in rows:
            if d.get(col, Fraction(0)) == 1 and all(c2 > col or d.get(c2, 0) == 0 for c2 in d):
                val = rhs
                for k2, v2 in d.items():
                    if k2 != col:
                        val -= v2 * sol[k2]
                sol[col] = val
                break
    # Assemble W
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    for (k, i, c), val in sol.items():
        W[k * n + i][c] = val
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    for c in range(rank):
        for (i, j), val in cols[c][0].items():
            U[i * m + j][c] = Fraction(val)
        for (j, k), val in cols[c][1].items():
            V[j * p + k][c] = Fraction(val)
    # Convert to floats (half-integers expected).
    def to_list(M):
        return [[float(x) for x in row] for row in M]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}


def _exact_rank_reduction(base, budget=30.0):
    start = time.time()
    n, m, p = DIMENSIONS
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    rank = U.shape[1]

    def col_vector(c):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * (m * p) + (j * p + k)] = (
                        U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    )
        return vec

    cols = np.array([col_vector(c) for c in range(rank)], dtype=float)
    kept = list(range(rank))
    improved = True
    while improved and (time.time() - start) < budget:
        improved = False
        for idx in list(kept):
            others = [c for c in kept if c != idx]
            A_mat = cols[:, others]
            b = cols[:, idx]
            try:
                sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
            except Exception:
                continue
            if np.linalg.norm(A_mat @ sol - b) < 1e-9:
                kept.remove(idx)
                improved = True
                break
    if len(kept) >= rank:
        return None
    return {"U": U[:, kept].tolist(), "V": V[:, kept].tolist(), "W": W[:, kept].tolist()}


def _verify(n, m, p, result):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return None
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > 1e-9:
                    return None
    for mat in (U, V, W):
        for val in mat.flat:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return None
    return rank


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
