"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Combined approach:
  * STRUCTURAL IDEA (from inspiration 2's factor framing): a rank-r decomposition
    is a multiset of r pairs (u_c, v_c) with u_c in {0,+-1,+-1/2}^m and
    v_c in {0,+-1,+-1/2}^p, and W[k*n+i, c] = u_c[i]*v_c[k] (derived, so the
    decomposition is exact by construction once the consistency condition holds).
  * REFINEMENT MACHINERY (from current program): exact integer/half-integer
    verification and a final polish pass that zeros redundant columns.

The consistency condition "u_c[i]*v_c[k] independent of j" is checked directly.
The coverage condition "sum_c u_c[i]*v_c[k] == 1 for all (i,k)" is the
optimization target. We solve it with greedy coverage + exact backtracking,
searching ranks from the lower bound upward.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 135.0  # safety margin before the 150s kill


def _build_schoolbook():
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


def _validate_exact(U, V, W, n, m, p):
    """Exact check that U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c
    using integer (doubled) arithmetic, so half-integers become integers."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True


def _to_list(M):
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result


def _enumerate_pairs(n, m, p):
    """Enumerate all consistent (u, v, w) patterns as integer (doubled) values.

    u: length m, each in {-2,-1,0,1,2}  (i.e. -1,-1/2,0,1/2,1)
    v: length p, each in {-2,-1,0,1,2}
    Consistent iff u[j]*v[j] is the same for all j (the value w2, in /4 units).
    Returns list of (u_tuple, v_tuple, w2).
    """
    vals = [-2, -1, 0, 1, 2]
    pairs = []
    seen = set()
    for u in np.ndindex(*([len(vals)] * m)):
        u_t = tuple(vals[x] for x in u)
        for v in np.ndindex(*([len(vals)] * p)):
            v_t = tuple(vals[x] for x in v)
            prods = [u_t[j] * v_t[j] for j in range(m)]
            if len(set(prods)) != 1:
                continue
            w2 = prods[0]
            key = (u_t, v_t)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((u_t, v_t, w2))
    return pairs


def _try_rank_greedy_bt(r, n, m, p, pairs, seed=0, max_iters=400):
    """Try to cover all (i,k) with exactly r consistent pairs.

    Greedy coverage to build an initial multiset, then exact backtracking that
    fixes the coverage to sum to 1 at every (i,k). Returns (U,V,W) or None.
    """
    rng = np.random.default_rng(seed)
    target = np.ones(n * p, dtype=int)  # 1 for every (i,k)

    for it in range(max_iters):
        if time.time() > DEADLINE - 3:
            return None

        # Greedy: pick pairs that cover the most currently-uncovered cells.
        chosen = []
        cov = np.zeros(n * p, dtype=int)
        for _ in range(r):
            if time.time() > DEADLINE - 3:
                return None
            best = None
            best_gain = -1
            # Candidate pool: prefer pairs with w2 == 4 (the "1" pattern).
            order = list(range(len(pairs)))
            rng.shuffle(order)
            for idx in order:
                u_t, v_t, w2 = pairs[idx]
                gain = 0
                for i in range(n):
                    for k in range(p):
                        if cov[i * p + k] == 0 and w2 != 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best = idx
                if gain >= n * p:
                    break
            if best is None:
                break
            u_t, v_t, w2 = pairs[best]
            chosen.append(best)
            for i in range(n):
                for k in range(p):
                    cov[i * p + k] += (u_t[i] * v_t[k])

        # Exact backtracking: adjust the multiset so cov == target everywhere.
        # Represent state as counts over the pair pool.
        counts = {}
        for idx in chosen:
            counts[idx] = counts.get(idx, 0) + 1
        cur = cov.copy()

        # Precompute contribution vector per pair.
        contrib = []
        for u_t, v_t, w2 in pairs:
            vec = np.zeros(n * p, dtype=int)
            for i in range(n):
                for k in range(p):
                    vec[i * p + k] = u_t[i] * v_t[k]
            contrib.append(vec)

        def bt(depth):
            if time.time() > DEADLINE - 3:
                return False
            if depth == 0:
                return np.array_equal(cur, target)
            # Which cells are wrong?
            diff = cur - target
            if np.all(diff == 0):
                return depth == 0
            # Pick the most-violated cell to guide the move.
            idx_cell = int(np.argmax(np.abs(diff)))
            need = -diff[idx_cell]  # we want to add 'need' to this cell
            # Try adding a pair that increases this cell by exactly 'need',
            # or removing a pair that decreases it.
            order = list(range(len(pairs)))
            rng.shuffle(order)
            for idx in order:
                vec = contrib[idx]
                # Add this pair (if it helps this cell)
                if vec[idx_cell] == need and need != 0:
                    for i in range(n):
                        for k in range(p):
                            cur[i * p + k] += vec[i * p + k]
                    counts[idx] = counts.get(idx, 0) + 1
                    if bt(depth - 1):
                        return True
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        del counts[idx]
                    for i in range(n):
                        for k in range(p):
                            cur[i * p + k] -= vec[i * p + k]
                # Remove this pair (if it helps this cell)
                if vec[idx_cell] == -need and need != 0 and counts.get(idx, 0) > 0:
                    for i in range(n):
                        for k in range(p):
                            cur[i * p + k] -= vec[i * p + k]
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        del counts[idx]
                    if bt(depth - 1):
                        return True
                    counts[idx] = counts.get(idx, 0) + 1
                    for i in range(n):
                        for k in range(p):
                            cur[i * p + k] += vec[i * p + k]
            return False

        # Seed the backtrack with a small number of free moves on top of greedy.
        # We allow up to r extra moves beyond the greedy fill to repair.
        if bt(2 * r):
            # Build U, V, W from counts.
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            W = np.zeros((p * n, r), dtype=int)
            col = 0
            for idx, cnt in counts.items():
                u_t, v_t, w2 = pairs[idx]
                for _ in range(cnt):
                    for j in range(m):
                        U[i * m + j, col] = u_t[j]
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, col] = v_t[k]
                    for i in range(n):
                        for k in range(p):
                            W[k * n + i, col] = w2
                    col += 1
            if col != r:
                # Truncate / pad to exactly r columns
                U = U[:, :r]
                V = V[:, :r]
                W = W[:, :r]
            # Convert to half-integers
            Uf = U / 2.0
            Vf = V / 2.0
            Wf = W / 4.0
            if _validate_exact(Uf, Vf, Wf, n, m, p):
                return {"U": _to_list(Uf), "V": _to_list(Vf), "W": _to_list(Wf)}
    return None


def _polish_reduce_rank(result, n, m, p, max_rounds=5):
    """Zero out columns that are linearly redundant, re-validating each time."""
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    r = U.shape[1]
    orig_len = len(result["U"][0])

    for _ in range(max_rounds):
        if time.time() > DEADLINE - 3:
            break
        removed = False
        for c in range(r):
            if time.time() > DEADLINE - 3:
                break
            if not np.any(U[:, c] != 0) or not np.any(V[:, c] != 0):
                U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
                U2[:, c] = 0; V2[:, c] = 0; W2[:, c] = 0
                if _validate_exact(U2, V2, W2, n, m, p):
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    r -= 1
                    removed = True
                    break
        if not removed:
            break

    if r < orig_len:
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    return result


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p

    lower = max(n * p, m * p, n * m)  # 24
    pairs = _enumerate_pairs(n, m, p)

    for r in range(lower, best_rank):
        if time.time() > DEADLINE - 5:
            break
        seed_base = r * 7919
        res = _try_rank_greedy_bt(r, n, m, p, pairs, seed=seed_base, max_iters=300)
        if res is not None:
            best = res
            best_rank = r
            polished = _polish_reduce_rank(best, n, m, p)
            if len(polished["U"][0]) < best_rank:
                best = polished
                best_rank = len(polished["U"][0])
            break

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
