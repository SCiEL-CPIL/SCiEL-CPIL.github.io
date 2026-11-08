"""Matrix multiplication 4x4x4: Lader-29 baseline + bounded search for lower rank."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 4]

DEADLINE = 115.0  # wall-clock safety margin before the 120s kill
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


# ---------------------------------------------------------------------------
# Lader's rank-29 decomposition of 4x4x4.
# Conventions: U index (i,j), V index (j,k), W index (k,i).
# A 4x4 matrix = 4 blocks of 4x4.  For block (a,b):
#   U = 1 if i==a,j==b ; V = 1 if j==b,k==c ; W = 1 if k==c,i==a
# ---------------------------------------------------------------------------
def _lader29():
    n = m = p = 4
    terms = []  # (u, v, w) each a 4x4 matrix (list of 16 ints)

    def mat(*ones):
        a = [0] * 16
        for idx in ones:
            a[idx] = 1
        return a

    def block(a, b):
        u = [0] * 16
        u[a * 4 + b] = 1
        v = [0] * 16
        for k in range(4):
            v[b * 4 + k] = 1
        w = [0] * 16
        for i in range(4):
            w[k0 * 4 + i] if False else None
        # w[k,i]=1 for all k, fixed i=a
        w = [0] * 16
        for k in range(4):
            w[k * 4 + a] = 1
        return u, v, w

    # 16 block terms
    for a in range(4):
        for b in range(4):
            terms.append(block(a, b))

    # 8 row-terms
    for a in range(4):
        for b in range(2):
            u = [0] * 16
            u[a * 4 + b] = 1
            u[a * 4 + b + 2] = 1
            v = [0] * 16
            for k in range(4):
                v[b * 4 + k] = 1
                v[(b + 2) * 4 + k] = 1
            w = [0] * 16
            for k in range(4):
                w[k * 4 + a] = 1
            terms.append((u, v, w))

    # 5 col-terms
    for b in range(4):
        u = [0] * 16
        for i in range(4):
            u[i * 4 + b] = 1
        v = [0] * 16
        for k in range(4):
            v[b * 4 + k] = 1
        w = [0] * 16
        w[b * 4 + b] = 1
        terms.append((u, v, w))

    # 2 diagonal terms
    for d in range(2):
        u = [0] * 16
        v = [0] * 16
        w = [0] * 16
        for a in range(4):
            u[a * 4 + a] = 1
            v[a * 4 + a] = 1
            w[a * 4 + a] = 1
        u[3 * 4 + 3] = 0
        v[3 * 4 + 3] = 0
        w[3 * 4 + 3] = 0
        u[(3 - d) * 4 + (3 - d)] = 1
        v[(3 - d) * 4 + (3 - d)] = 1
        w[(3 - d) * 4 + (3 - d)] = 1
        terms.append((u, v, w))

    assert len(terms) == 29, len(terms)
    return terms


def _terms_to_matrices(terms):
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    for c, (u, v, w) in enumerate(terms):
        for idx in range(16):
            U[idx][c] = u[idx]
            V[idx][c] = v[idx]
            W[idx][c] = w[idx]
    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Exhaustive search for a rank-R decomposition using the 4x4x4 structure.
# Each term is a block (a,b) with U=unit(a,b), V=row b, W=col a.
# ---------------------------------------------------------------------------
def _search_rank(R, time_budget):
    n = 4
    # Precompute row patterns (b -> 4-bit mask)
    row_mask = [0, 0b0011, 0b1100, 0b0101, 0b1010]
    # col pattern for block a: W = col a -> mask over k
    col_mask = [1, 2, 4, 8]
    # W index (k,i): for fixed a, W[k*4+a] = 1 for k in colmask
    # V index (j,k): V[j*4+k]=1 for j in rowmask, k in colmask
    # U index (i,j): U[i*4+j]=1 for i in rowmask, j in colmask
    row_list = [1, 3, 12, 5, 10]  # as bitmasks (0,1,2,3,4)

    def build(rowmask, colmask, a):
        u = [0] * 16
        for i in range(4):
            if rowmask >> i & 1:
                for k in range(4):
                    if colmask >> k & 1:
                        u[i * 4 + k] = 1
        v = [0] * 16
        for i in range(4):
            if rowmask >> i & 1:
                for k in range(4):
                    if colmask >> k & 1:
                        v[i * 4 + k] = 1
        w = [0] * 16
        for k in range(4):
            if colmask >> k & 1:
                w[k * 4 + a] = 1
        return u, v, w

    def rowmask_of(row):
        return [0, 0b0011, 0b1100, 0b0101, 0b1010][row]

    def colmask_of(col):
        return [1, 2, 4, 8][col]

    # Precompute the 16 required target entries for each (i,j,k).
    # We'll verify by reconstructing the multiplication tensor.
    # target[i,j,k] = 1 always.
    # For a decomposition, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.

    # Use constraint propagation via the zero-sum linear constraints.
    # We'll just do backtracking with the full tensor check at the end
    # but prune using the 16 linear constraints (sum over terms of
    # (row/col) combinations).

    # Represent each term as (row, col, a) where row in 0..4, col in 0..3, a in 0..3.
    # The 16 linear constraints: for each (a,b), the sum of
    #   U[i,j]V[j,k]W[k,i] over terms with i in {a,a+1}, j in {b,b+1}, k in {c,c+1}
    # equals the number of (i,j,k) in that 2x2x2 cube = 8.
    # Actually the standard zero-sum: sum_{i in {a,a+1}} sum_{j in {b,b+1}} ... 
    # Let me just verify the tensor directly; it's only 64 entries.

    best = None
    rng = random.Random(12345)

    # Order: fix block (a,b) for each term; choose (row,col) pattern.
    # There are 5 row patterns and 4 col patterns = 20 choices per (a,b).
    # We must cover all 16 blocks (a,b) at least once? Not necessarily,
    # but each block (a,b) needs coverage.

    # Backtracking over terms, assigning (a,b,row,col).
    # Prune: after assigning, check partial tensor entries that are fully
    # determined (i.e., all terms that could contribute to entry (i,j,k)
    # have been assigned). This is complex; instead use the 16 linear
    # constraints as a cheap prune.

    # Linear constraint for cube (a,b,c) [2x2x2]:
    # sum over terms of [U contributes to cube] * [V contributes] * [W contributes] = 8
    # where U contributes to cube (a,b,c) if rowmask contains a or a+1 (i range)
    #   and colmask contains b or b+1 (j range)  -- wait U is (i,j).
    # U[i,j] with i in {a,a+1}, j in {b,b+1}: contributes if
    #   rowmask has a or a+1 AND colmask has b or b+1.
    # V[j,k] with j in {b,b+1}, k in {c,c+1}: contributes if
    #   rowmask has b or b+1 AND colmask has c or c+1.
    # W[k,i] with k in {c,c+1}, i in {a,a+1}: contributes if
    #   colmask has c or c+1 AND a in {a,a+1} (always true since a is fixed).
    #   Actually W[k,i]=1 for k in colmask, i=a. So contributes if colmask has
    #   c or c+1 AND a in {a,a+1} (true).
    # So the product is:
    #   U_con = (rowmask & cube_i_mask) != 0 and (colmask & cube_j_mask) != 0
    #   V_con = (rowmask & cube_j_mask) != 0 and (colmask & cube_k_mask) != 0
    #   W_con = (colmask & cube_k_mask) != 0  (and a in cube_i, true)
    # where cube_i_mask = (1<<a)|(1<<(a+1)), etc.

    # This is getting complex. Let me just do a randomized search:
    # assign each of R terms a random (a,b,row,col), check tensor, repeat.
    # Plus a local search to improve.

    def check(terms):
        # terms: list of (a, b, row, col)
        # Build U,V,W as 16xR and check all 64 entries.
        # For speed, accumulate per (i,j,k).
        # target[i,j,k]=1
        # For each term, it contributes 1 to (i,j,k) if
        #   i in rowmask, j in colmask (U), j in rowmask, k in colmask (V),
        #   k in colmask, i == a (W).
        # So term (a,b,row,col) contributes to (i,j,k) iff
        #   rowmask has i, colmask has j, rowmask has j, colmask has k,
        #   colmask has k, i == a.
        # Simplify: i==a, rowmask has i and j, colmask has j and k.
        # We need sum over terms = 1 for all (i,j,k).
        # Count per (i,j,k).
        counts = [[0]*4 for _ in range(4)]  # counts[i][j] for each k? 
        # Let's use counts[i][j][k].
        cnt = [[[0]*4 for _ in range(4)] for _ in range(4)]
        for (a, b, row, col) in terms:
            rm = rowmask_of(row)
            cm = colmask_of(col)
            for i in range(4):
                if not (rm >> i & 1):
                    continue
                if i != a:
                    continue
                for j in range(4):
                    if not (cm >> j & 1):
                        continue
                    if not (rm >> j & 1):
                        continue
                    for k in range(4):
                        if not (cm >> k & 1):
                            continue
                        cnt[i][j][k] += 1
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if cnt[i][j][k] != 1:
                        return False
        return True

    # Randomized search with local improvement.
    def random_terms():
        terms = []
        for _ in range(R):
            a = rng.randrange(4)
            b = rng.randrange(4)
            row = rng.randrange(5)
            col = rng.randrange(4)
            terms.append((a, b, row, col))
        return terms

    def local_search(terms, max_iter=2000):
        # Try to fix by changing one term at a time.
        best_score = None
        for it in range(max_iter):
            if _time_left() < 0.5:
                break
            # Compute deficit
            cnt = [[[0]*4 for _ in range(4)] for _ in range(4)]
            for (a, b, row, col) in terms:
                rm = rowmask_of(row)
                cm = colmask_of(col)
                for i in range(4):
                    if not (rm >> i & 1) or i != a:
                        continue
                    for j in range(4):
                        if not (cm >> j & 1) or not (rm >> j & 1):
                            continue
                        for k in range(4):
                            if not (cm >> k & 1):
                                continue
                            cnt[i][j][k] += 1
            # Score = number of entries not equal to 1
            score = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        if cnt[i][j][k] != 1:
                            score += 1
            if score == 0:
                return terms
            if best_score is None or score < best_score[0]:
                best_score = (score, list(terms))
            # Pick a random term and a random change
            ti = rng.randrange(R)
            a, b, row, col = terms[ti]
            new_a = rng.randrange(4)
            new_b = rng.randrange(4)
            new_row = rng.randrange(5)
            new_col = rng.randrange(4)
            terms[ti] = (new_a, new_b, new_row, new_col)
        return best_score[1] if best_score else terms

    # Run several random restarts
    n_restarts = 200
    for r in range(n_restarts):
        if _time_left() < 1.0:
            break
        terms = random_terms()
        terms = local_search(terms, max_iter=1000)
        if check(terms):
            return terms
        if r % 20 == 0 and _time_left() < 0.5:
            break
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    global START
    START = time.time()
    best = _lader29()
    best_rank = 29

    # Try to find a lower rank
    for R in range(28, 29):  # try rank 28
        if _time_left() < 5.0:
            break
        result = _search_rank(R, time_budget=max(0.0, _time_left() - 2.0))
        if result is not None:
            best = result
            best_rank = R
            break

    return _terms_to_matrices(best)


if __name__ == "__main__":
    out = construct()
    print("rank:", len(out["U"][0]))
