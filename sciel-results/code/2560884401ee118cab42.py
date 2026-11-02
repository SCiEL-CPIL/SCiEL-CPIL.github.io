"""Matrix multiplication 2x2x15 tensor decomposition (rank-minimal exact).

Design family: exploit the tensor's block structure.  The 2x2x15 tensor
decomposes into independent sub-tensors over groups of k-indices.  For each
group size g in {3,4,5} we try to find a small-rank exact decomposition of the
2x2xg block and replicate it across the p/g groups.  A rank-R block gives total
rank R*(p/g).  We also run a direct full-tensor exact search as a fallback.

The block solver is exact (rational arithmetic) and fast: for each candidate W
support pattern we solve a small linear system for U and V (linear once the
other factor is fixed) and verify exactness.  We keep an explicit deadline guard
and always return the best valid decomposition found (falling back to a
rank-53 Strassen construction).
"""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import combinations, product

DIMENSIONS = [2, 2, 15]


# --------------------------------------------------------------------------- #
# Small exact linear algebra over Fraction
# --------------------------------------------------------------------------- #
def _det(M):
    n = len(M)
    if n == 0:
        return Fraction(1)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    M = [row[:] for row in M]
    sign = Fraction(1)
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            sign = -sign
        pv = M[col][col]
        for r in range(col + 1, n):
            if M[r][col] != 0:
                factor = M[r][col] / pv
                for c in range(col, n):
                    M[r][c] -= factor * M[col][c]
    d = sign
    for r in range(n):
        d *= M[r][r]
    return d


def _solve_linear(M, b):
    """Solve M x = b for a square invertible M (Fraction). Returns x or None."""
    n = len(M)
    M2 = [row[:] for row in M]
    b2 = b[:]
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if M2[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != col:
            M2[col], M2[pivot] = M2[pivot], M2[col]
            b2[col], b2[pivot] = b2[pivot], b2[col]
        pv = M2[col][col]
        for r in range(n):
            if r == col:
                continue
            if M2[r][col] != 0:
                factor = M2[r][col] / pv
                for c in range(col, n):
                    M2[r][c] -= factor * M2[col][c]
                b2[r] -= factor * b2[col]
    return [b2[r] / M2[r][r] for r in range(n)]


def _pick_invertible_columns(A, rows, max_combos=200000):
    """Pick `rows` columns of A (rows x R) forming an invertible square submatrix."""
    R = len(A[0])
    if R < rows:
        return None
    count = 0
    for combo in combinations(range(R), rows):
        count += 1
        if count > max_combos:
            break
        M = [[A[i][c] for c in combo] for i in range(rows)]
        if _det(M) != 0:
            return list(combo)
    return None


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #
def _is_half_int(fr):
    return fr.denominator in (1, 2)


def _verify_int(U, V, W, n, m, p):
    """Verify exact integer decomposition: every (i,j,k) entry equals 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True


def _verify_2x2xg(U, V, W, n, m, g, tol=1e-9):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(g):
                actual = sum(U[i * m + j, r] * V[j * g + k, r] * W[k * n + i, r]
                             for r in range(R))
                if abs(actual - 1.0) > tol:
                    return False
    return True


# --------------------------------------------------------------------------- #
# Block decomposition solver (exact, rational)
# --------------------------------------------------------------------------- #
def _solve_block(n, m, g, R, W_supports, deadline):
    """Given a list of R W-support dicts (each {(k,i): coeff}), solve for U, V.

    Returns (Uf, Vf, Wf) as float arrays if an exact half-integer solution is
    found, else None.
    """
    # Build W as a Fraction matrix (g*n x R).
    Wm = [[Fraction(0)] * R for _ in range(g * n)]
    for r in range(R):
        for (k, i), c in W_supports[r].items():
            Wm[k * n + i][r] = Fraction(c)

    # Initialize V with a sparse selector pattern.
    Vm = [[Fraction(0)] * R for _ in range(m * g)]
    for r in range(R):
        Vm[r % (m * g)][r] = Fraction(1)

    Um = [[Fraction(0)] * R for _ in range(n * m)]

    for _ in range(8):
        # ---- Solve U given V, W ----
        # For each (i,j): unknowns U[i,j,r] (R). Equations over k:
        #   sum_r U[i,j,r] * V[j*3+k, r] * W[k*n+i, r] = 1   (g equations)
        solved_U = True
        for i in range(n):
            for j in range(m):
                A = [[Fraction(0)] * R for _ in range(g)]
                for k in range(g):
                    row = A[k]
                    Vrow = Vm[j * g + k]
                    for r in range(R):
                        wv = Wm[k * n + i][r]
                        if wv != 0 and Vrow[r] != 0:
                            row[r] = Vrow[r] * wv
                cols = _pick_invertible_columns(A, g)
                if cols is None:
                    solved_U = False
                    break
                M = [[A[k][c] for c in cols] for k in range(g)]
                b = [Fraction(1) for _ in range(g)]
                x = _solve_linear(M, b)
                if x is None:
                    solved_U = False
                    break
                Urow = Um[i * m + j]
                for r in range(R):
                    Urow[r] = Fraction(0)
                for r, val in zip(cols, x):
                    Urow[r] = val
        if not solved_U:
            return None

        # ---- Solve V given U, W ----
        # For each (j,k): unknowns V[j,k,r] (R). Equations over i:
        #   sum_r U[i*m+j, r] * V[j*g+k, r] * W[k*n+i, r] = 1   (n equations)
        solved_V = True
        for j in range(m):
            for k in range(g):
                A = [[Fraction(0)] * R for _ in range(n)]
                for i in range(n):
                    row = A[i]
                    Urow = Um[i * m + j]
                    for r in range(R):
                        wv = Wm[k * n + i][r]
                        if wv != 0 and Urow[r] != 0:
                            row[r] = Urow[r] * wv
                cols = _pick_invertible_columns(A, n)
                if cols is None:
                    solved_V = False
                    break
                M = [[A[i][c] for c in cols] for i in range(n)]
                b = [Fraction(1) for _ in range(n)]
                x = _solve_linear(M, b)
                if x is None:
                    solved_V = False
                    break
                Vrow = Vm[j * g + k]
                for r in range(R):
                    Vrow[r] = Fraction(0)
                for r, val in zip(cols, x):
                    Vrow[r] = val
        if not solved_V:
            return None

        # ---- Check exactness + half-integer ----
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(g):
                    s = Fraction(0)
                    Urow = Um[i * m + j]
                    Vrow = Vm[j * g + k]
                    for r in range(R):
                        s += Urow[r] * Vrow[r] * Wm[k * n + i][r]
                    if s != Fraction(1):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue  # iterate again (alternation may converge)

        # Half-integer check.
        for mat in (Um, Vm, Wm):
            for row in mat:
                for val in row:
                    if not _is_half_int(val):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return None

        # Convert to float.
        def to_float(mat):
            return np.array([[float(v) for v in row] for row in mat], dtype=float)
        Uf, Vf, Wf = to_float(Um), to_float(Vm), to_float(Wm)
        if _verify_2x2xg(Uf, Vf, Wf, n, m, g):
            return Uf, Vf, Wf
        return None


def _build_support_candidates(g, R, rng, max_random):
    """Build a list of candidate W-support patterns (list of R dicts)."""
    ki = [(k, i) for k in range(g) for i in range(2)]
    singles = []
    for pos in ki:
        singles.append({pos: 1})
        singles.append({pos: -1})
    doubles = []
    for a, b in product(ki, repeat=2):
        if a == b:
            continue
        for ca in (1, -1):
            for cb in (1, -1):
                doubles.append({a: ca, b: cb})

    candidates = []
    # Structured: single-position cycling with offsets.
    for off in range(len(ki)):
        candidates.append([ {ki[(r + off) % len(ki)]: 1} for r in range(R) ])
    # Structured: double-position cycling.
    for off in range(len(doubles)):
        candidates.append([ doubles[(r + off) % len(doubles)] for r in range(R) ])
    # Randomized.
    pool = singles + doubles
    for _ in range(max_random):
        candidates.append([ rng.choice(pool) for r in range(R) ])
    return candidates


def _find_block(n, m, g, R, deadline):
    """Find an exact rank-R decomposition of the 2x2xg block, or None."""
    rng = random.Random(1000 + g * 100 + R)
    # Budget random candidates by remaining time; start modest, scale up.
    remaining = max(0.0, deadline - time.time())
    max_random = int(300 * remaining)
    max_random = max(200, min(max_random, 200000))
    candidates = _build_support_candidates(g, R, rng, max_random)
    for supports in candidates:
        if time.time() > deadline - 1.0:
            break
        res = _solve_block(n, m, g, R, supports, deadline)
        if res is not None:
            return res
    return None


def _replicate_block(n, m, p, g, U10, V10, W10):
    """Replicate a 2x2xg block decomposition across the p/g groups of k."""
    R = U10.shape[1]
    t = p // g
    total_rank = R * t
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for grp in range(t):
        base = grp * R
        k0 = grp * g
        for r in range(R):
            c = base + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U10[i * m + j, r]
            for j in range(m):
                for kl in range(g):
                    V[j * p + (k0 + kl)][c] = V10[j * g + kl, r]
            for kl in range(g):
                for i in range(n):
                    W[(k0 + kl) * n + i][c] = W10[kl * n + i, r]
    return U, V, W


# --------------------------------------------------------------------------- #
# Direct full-tensor exact search (fallback)
# --------------------------------------------------------------------------- #
def _find_full(n, m, p, R, deadline):
    """Direct exact rank-R decomposition of the full 2x2x15 tensor, or None."""
    rng = random.Random(7777 + R)
    ki = [(k, i) for k in range(p) for i in range(n)]
    singles = []
    for pos in ki:
        singles.append({pos: 1})
        singles.append({pos: -1})
    remaining = max(0.0, deadline - time.time())
    max_random = int(60 * remaining)
    max_random = max(150, min(max_random, 60000))
    for _ in range(max_random):
        if time.time() > deadline - 1.0:
            break
        supports = [ rng.choice(singles) for r in range(R) ]
        res = _solve_block(n, m, p, R, supports, deadline)
        if res is not None:
            return res
    return None


# --------------------------------------------------------------------------- #
# Fallback constructions (always valid)
# --------------------------------------------------------------------------- #
def _strassen_terms(k0, k1):
    """Standard Strassen rank-7 terms for a 2x2x2 block (k indices k0, k1)."""
    return [
        ({(0, 0): 1, (1, 1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, k0): 1}, {(k0, 1): 1, (k1, 1): -1}),
        ({(0, 0): 1}, {(0, k1): 1, (1, k1): -1}, {(k1, 0): 1, (k1, 1): 1}),
        ({(1, 1): 1}, {(1, k0): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for a 2x2x1 block."""
    return [
        ({(0, 0): 1}, {(0, k): 1}, {(k, 0): 1}),
        ({(0, 1): 1}, {(1, k): 1}, {(k, 0): 1}),
        ({(1, 0): 1}, {(0, k): 1}, {(k, 1): 1}),
        ({(1, 1): 1}, {(1, k): 1}, {(k, 1): 1}),
    ]


def _build_strassen_53(n, m, p):
    """Rank-53: seven 2x2x2 Strassen blocks (rank 7) + one 2x2x1 schoolbook (rank 4)."""
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14,)]
    all_U, all_V, all_W = [], [], []
    for block in blocks:
        if len(block) == 2:
            terms = _strassen_terms(block[0], block[1])
        else:
            terms = _schoolbook_terms(block[0])
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k_idx, c) for (j, k_idx), c in v_dict.items()]
            w_entries = [(k_idx * n + i, c) for (k_idx, i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val
    return U, V, W


def _build_schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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
    return U, V, W


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # safety margin under the 150s kill

    # Baseline: rank-53 Strassen construction (always valid).
    U53, V53, W53 = _build_strassen_53(n, m, p)
    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try block decompositions for group sizes that divide 15: 3, 5.
    # For each, search block ranks from low to high; the first valid one for a
    # group size gives total rank R*(p/g).  We keep the overall minimum.
    for g in (3, 5):
        if time.time() > deadline - 30:
            break
        t = p // g
        # Target total ranks to beat: try block ranks giving total < best_rank.
        # Block rank R gives total R*t.  Search R from a low bound upward.
        R_min = 1
        R_max = (best_rank - 1) // t
        for R in range(R_min, R_max + 1):
            if time.time() > deadline - 25:
                break
            block = _find_block(n, m, g, R, deadline)
            if block is None:
                continue
            U10, V10, W10 = block
            U, V, W = _replicate_block(n, m, p, g, U10, V10, W10)
            if _verify_int(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                break  # smallest R for this g is best for this g

    # Direct full-tensor search for any rank below the current best.
    if time.time() < deadline - 20:
        for R in range(1, best_rank):
            if time.time() > deadline - 15:
                break
            res = _find_full(n, m, p, R, deadline)
            if res is None:
                continue
            Uf, Vf, Wf = res
            U = [[int(round(x)) for x in row] for row in np.round(Uf * 2) / 2]
            V = [[int(round(x)) for x in row] for row in np.round(Vf * 2) / 2]
            W = [[int(round(x)) for x in row] for row in np.round(Wf * 2) / 2]
            if _verify_int(U, V, W, n, m, p):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break

    return {"U": best_U, "V": best_V, "W": best_W}
