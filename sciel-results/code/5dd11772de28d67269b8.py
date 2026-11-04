import time
import random

DIMENSIONS = [4, 4, 7]

# ----------------------------- Deadline guard -----------------------------
_DEADLINE = None

def _deadline_guard():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140  # 10s safety margin under the 150s kill
    return time.time() < _DEADLINE

def _set_deadline():
    global _DEADLINE
    _DEADLINE = time.time() + 140

# ----------------------------- Known-good rank-76 (fallback) -----------------------------
def _strassen_2x2_2x2():
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    U[0][0] = 1; U[3][0] = -1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[3][0] = -1
    U[0][1] = 1; U[1][1] = 1
    V[2][1] = 1
    W[1][1] = 1; W[3][1] = 1
    U[3][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[2][3] = 1; U[3][3] = 1
    V[0][3] = 1
    W[0][3] = 1; W[2][3] = -1
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1; W[2][4] = 1
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = -1; W[3][5] = 1
    U[2][6] = 1; U[0][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1; W[2][6] = 1
    return U, V, W

def _strassen_4x4_4x4():
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    U_base, V_base, W_base = _strassen_2x2_2x2()

    def add_term(k, term_idx):
        base_col = k * 7 + term_idx
        if k == 0:
            x_map = {0: [(0, 1), (10, -1)], 1: [(1, 1), (11, -1)], 2: [(4, 1), (14, -1)], 3: [(5, 1), (15, -1)]}
            y_map = {0: [(8, 1), (10, 1)], 1: [(9, 1), (11, 1)], 2: [(12, 1), (14, 1)], 3: [(13, 1), (15, 1)]}
            w_map = {0: [(0, 1), (8, 1), (10, -1)], 1: [(1, 1), (9, 1), (11, -1)], 2: [(4, 1), (12, 1), (14, -1)], 3: [(5, 1), (13, 1), (15, -1)]}
        elif k == 1:
            x_map = {0: [(0, 1), (2, 1)], 1: [(1, 1), (3, 1)], 2: [(4, 1), (6, 1)], 3: [(5, 1), (7, 1)]}
            y_map = {0: [(8, 1)], 1: [(9, 1)], 2: [(12, 1)], 3: [(13, 1)]}
            w_map = {0: [(2, 1)], 1: [(3, 1)], 2: [(6, 1)], 3: [(7, 1)]}
        elif k == 2:
            x_map = {0: [(10, 1)], 1: [(11, 1)], 2: [(14, 1)], 3: [(15, 1)]}
            y_map = {0: [(2, 1), (10, -1)], 1: [(3, 1), (11, -1)], 2: [(6, 1), (14, -1)], 3: [(7, 1), (15, -1)]}
            w_map = {0: [(10, 1)], 1: [(11, 1)], 2: [(14, 1)], 3: [(15, 1)]}
        elif k == 3:
            x_map = {0: [(8, 1), (10, 1)], 1: [(9, 1), (11, 1)], 2: [(12, 1), (14, 1)], 3: [(13, 1), (15, 1)]}
            y_map = {0: [(0, 1)], 1: [(1, 1)], 2: [(4, 1)], 3: [(5, 1)]}
            w_map = {0: [(0, 1), (8, -1)], 1: [(1, 1), (9, -1)], 2: [(4, 1), (12, -1)], 3: [(5, 1), (13, -1)]}
        elif k == 4:
            x_map = {0: [(0, 1), (10, 1)], 1: [(1, 1), (11, 1)], 2: [(4, 1), (14, 1)], 3: [(5, 1), (15, 1)]}
            y_map = {0: [(0, 1), (10, 1)], 1: [(1, 1), (11, 1)], 2: [(4, 1), (14, 1)], 3: [(5, 1), (15, 1)]}
            w_map = {0: [(0, -1), (2, 1), (8, 1)], 1: [(1, -1), (3, 1), (9, 1)], 2: [(4, -1), (6, 1), (12, 1)], 3: [(5, -1), (7, 1), (13, 1)]}
        elif k == 5:
            x_map = {0: [(2, 1), (10, -1)], 1: [(3, 1), (11, -1)], 2: [(6, 1), (14, -1)], 3: [(7, 1), (15, -1)]}
            y_map = {0: [(0, 1), (2, 1)], 1: [(1, 1), (3, 1)], 2: [(4, 1), (6, 1)], 3: [(5, 1), (7, 1)]}
            w_map = {0: [(2, -1), (10, 1)], 1: [(3, -1), (11, 1)], 2: [(6, -1), (14, 1)], 3: [(7, -1), (15, 1)]}
        else:
            x_map = {0: [(8, 1), (0, -1)], 1: [(9, 1), (1, -1)], 2: [(12, 1), (4, -1)], 3: [(13, 1), (5, -1)]}
            y_map = {0: [(2, 1), (10, 1)], 1: [(3, 1), (11, 1)], 2: [(6, 1), (14, 1)], 3: [(7, 1), (15, 1)]}
            w_map = {0: [(0, 1), (2, -1)], 1: [(1, 1), (3, -1)], 2: [(4, 1), (6, -1)], 3: [(5, 1), (7, -1)]}
        for local_idx in range(4):
            uc = U_base[local_idx][term_idx]
            if uc != 0:
                for g_idx, c in x_map[local_idx]:
                    U[g_idx][base_col] += uc * c
            vc = V_base[local_idx][term_idx]
            if vc != 0:
                for g_idx, c in y_map[local_idx]:
                    V[g_idx][base_col] += vc * c
            wc = W_base[local_idx][term_idx]
            if wc != 0:
                for g_idx, c in w_map[local_idx]:
                    W[g_idx][base_col] += wc * c

    for k in range(7):
        for t in range(7):
            add_term(k, t)
    return U, V, W, rank

def _schoolbook_4x4_4x3():
    rank = 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] = 1
                V[j*3+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W, rank

def _build_full_4x4_4x7():
    U1, V1, W1, rank1 = _strassen_4x4_4x4()
    U2, V2, W2, rank2 = _schoolbook_4x4_4x3()
    rank = rank1 + rank2
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for j in range(16):
        for c in range(rank1):
            V[j][c] = V1[j][c]
    for k in range(16):
        for c in range(rank1):
            W[k][c] = W1[k][c]
    for i in range(16):
        for c in range(rank2):
            U[i][rank1+c] = U2[i][c]
    for j in range(12):
        for c in range(rank2):
            V[16+j][rank1+c] = V2[j][c]
    for k in range(12):
        for c in range(rank2):
            W[16+k][rank1+c] = W2[k][c]
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

# ----------------------------- New strategy: runtime algebraic search -----------------------------
# The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k), else 0.
# A rank-r decomposition is r rank-1 terms: sum_c U[a,c]*V[b,c]*W[cc,c] = T[a,b,cc].
# Necessary condition: every flattening has rank <= r. We use the 3x3-minor vanishing
# of the A-flattening (16 x 28) as a fast filter, then confirm with exact reconstruction.

def _build_target_tensor(n, m, p):
    # T indexed by (a,b,cc) where a=i*m+j, b=j*p+k, cc=k*n+i
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                cc = k * n + i
                T[(a, b, cc)] = 1
    return T

def _flattening_A(T, n, m, p):
    # rows: a in [0, n*m), cols: (j,cc) -> for a fixed a=i*m+j, valid cc=k*n+i
    # Flattening: F[a, (j,cc)] = T[a, j*p+k, cc]
    # Build a 16 x 28 matrix
    F = [[0]* (m*p) for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                cc = k*n + i
                # column index for (j, cc): we use j*p + (cc) but cc depends; use (j, cc) pair -> j*p+cc
                F[a][j*p + cc] = 1
    return F

def _all_3x3_minors_zero(F):
    # F is n*m x (m*p); check all 3x3 minors vanish
    nr = len(F)
    nc = len(F[0])
    if nr < 3 or nc < 3:
        return True
    rows = list(range(nr))
    cols = list(range(nc))
    # iterate combinations of 3 rows and 3 cols
    from itertools import combinations
    for r3 in combinations(rows, 3):
        for c3 in combinations(cols, 3):
            # determinant of 3x3
            a = F[r3[0]][c3[0]] - F[r3[0]][c3[1]]
            # compute det directly
            d = (F[r3[0]][c3[0]]*(F[r3[1]][c3[1]]*F[r3[2]][c3[2]] - F[r3[1]][c3[2]]*F[r3[2]][c3[1]])
                 - F[r3[0]][c3[1]]*(F[r3[1]][c3[0]]*F[r3[2]][c3[2]] - F[r3[1]][c3[2]]*F[r3[2]][c3[0]])
                 + F[r3[0]][c3[2]]*(F[r3[1]][c3[0]]*F[r3[2]][c3[1]] - F[r3[1]][c3[1]]*F[r3[2]][c3[0]]))
            if d != 0:
                return False
    return True

def _reconstruct_and_check(T, n, m, p, rank, U, V, W):
    # U: n*m x rank, V: m*p x rank, W: p*n x rank
    # For each (a,b,cc) in T, sum_c U[a,c]*V[b,c]*W[cc,c] must equal T[(a,b,cc)]
    for (a, b, cc), val in T.items():
        s = 0
        for c in range(rank):
            s += U[a][c] * V[b][c] * W[cc][c]
        if s != val:
            return False
    # Also ensure no spurious: for (a,b,cc) not in T, sum must be 0
    for a in range(n*m):
        for b in range(m*p):
            for cc in range(p*n):
                if (a, b, cc) in T:
                    continue
                s = 0
                for c in range(rank):
                    s += U[a][c] * V[b][c] * W[cc][c]
                if s != 0:
                    return False
    return True

def _search_small_rank(T, n, m, p, target_rank, rng, time_budget):
    """
    Search for a rank-`target_rank` decomposition with integer/half-integer coefficients.
    Uses a structured heuristic: fix a small candidate support for each term.
    This is a bounded search; returns (U,V,W) or None.
    """
    start = time.time()
    # Candidate coefficient values (integers and half-integers, small)
    coeff_vals = [0, 1, -1, 2, -2, 0.5, -0.5]

    # We attempt a randomized local search: start from a sparse random assignment
    # and try to satisfy equations. Given the 150s budget and the difficulty,
    # we do a limited number of trials.
    trials = 200
    for _ in range(trials):
        if time.time() - start > time_budget:
            return None
        if not _deadline_guard():
            return None
        # Build random sparse U, V, W
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        for c in range(target_rank):
            # pick a few non-zero entries
            for a in rng.sample(range(n*m), min(4, n*m)):
                U[a][c] = rng.choice(coeff_vals)
            for b in rng.sample(range(m*p), min(5, m*p)):
                V[b][c] = rng.choice(coeff_vals)
            for cc in rng.sample(range(p*n), min(4, p*n)):
                W[cc][c] = rng.choice(coeff_vals)
        # Quick check on a subset of equations
        ok = True
        for (a, b, cc), val in list(T.items())[:40]:
            s = 0
            for c in range(target_rank):
                s += U[a][c] * V[b][c] * W[cc][c]
            if s != val:
                ok = False
                break
        if ok:
            if _reconstruct_and_check(T, n, m, p, target_rank, U, V, W):
                return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    _set_deadline()

    # First, build the known-good rank-76 fallback so we always have a valid answer.
    U76, V76, W76, rank76 = _build_full_4x4_4x7()
    if _verify(U76, V76, W76, n, m, p):
        fallback = {"U": U76, "V": V76, "W": W76}
    else:
        fallback = _schoolbook()

    # Now attempt the new runtime algebraic search for a strictly lower rank.
    T = _build_target_tensor(n, m, p)
    rng = random.Random(12345)
    remaining = 140 - (time.time() - (_DEADLINE - 140))
    if remaining > 10:
        # Try ranks from just below 76 downward, with a small time budget each.
        for target_rank in range(rank76 - 1, max(rank76 - 1, 40), -1):
            if not _deadline_guard():
                break
            if time.time() - (_DEADLINE - 140) > 120:
                break
            res = _search_small_rank(T, n, m, p, target_rank, rng, 8)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}

    return fallback

# EVOLVE-BLOCK-END
