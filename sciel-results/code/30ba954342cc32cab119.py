"""Rewritten QAP solver: multi-start ILS with fast swap-delta local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120 s
    rng = np.random.default_rng(12345)

    # Precompute row sums of A (constant per facility)
    a_row = a.sum(axis=1)

    # ---------- fast cost evaluation for a given permutation p ----------
    def full_cost(p):
        # sum a[i,j]*b[p[i],p[j]]
        return int(np.einsum('ij,kl->', a, b[np.ix_(p, p)]))

    # ---------- local search: 2-opt swaps with fast delta ----------
    def local_search(p):
        # p: int64 array, facility -> location
        # precompute per-facility contribution vector: g[i] = sum_j a[i,j]*b[p[i], p[j]]
        # but we need symmetric handling. Use delta formula for swapping facilities i,k.
        # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
        # swap locations of i and k: delta = (a[i,k]+a[k,i])*(b[p[i],p[k]] - b[p[k],p[i]])
        #   + sum_{j!=i,k} (a[i,j]+a[k,j])*(b[p[k],p[j]] - b[p[i],p[j]])  -- but this is O(n)
        # Use precomputed: for each facility i, row_i = a[i,:] ; for location loc, col of B
        # delta(i,k) = (a[i,k]+a[k,i])*(b[p[i],p[k]] - b[p[k],p[i]])
        #              + sum_{j} (a[i,j]-a[k,j])*(b[p[k],p[j]] - b[p[i],p[j]])
        # Precompute S_i = sum_j a[i,j]*b[p[i],p[j]]? Let's use:
        # Let r_i = a[i,:] (row). Define T_i = r_i @ B[:, p[i]]? Not directly.
        # Efficient: define for each facility i a vector v_i = a[i,:] ; then
        #   f_i(x) = v_i @ B[:, x]  (cost contribution of facility i to location x)
        # Then total cost = sum_i f_i(p[i])
        # Swapping i,k changes cost by:
        #   delta = f_i(p[k]) + f_k(p[i]) - f_i(p[i]) - f_k(p[k])
        # Precompute f_i for all i? That's n x n matrix F where F[i,x]=a[i,:]@B[:,x].
        # F = a @ B.T  (n x n). Then f_i(x) = F[i,x].
        # delta(i,k) = F[i,p[k]] + F[k,p[i]] - F[i,p[i]] - F[k,p[k]]
        # This is O(1) per swap!
        # But wait: does total cost = sum_i a[i,j] b[p[i],p[j]] = sum_i (a[i,:] @ B[:,p[i]])?
        # sum_i sum_j a[i,j] b[p[i],p[j]] = sum_i (a[i,:] dot B[:,p[i]])  YES.
        return _local_search_impl(p, F_cache, a)

    # Build F = a @ b.T  (n x n): F[i, x] = sum_j a[i,j]*b[j, x] = a[i,:] @ B[:,x]
    F_cache = a @ b.T  # note: b.T[x, j] = b[j, x]

    def _local_search_impl(p, F, a):
        n = len(p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # compute current cost
            cur = int(np.einsum('ij,j->', F, p))  # sum_i F[i, p[i]]
            # steepest descent: find best improving swap
            best_delta = 0
            best_i = best_k = -1
            # To keep it fast, do a randomized / partial scan
            # Full O(n^2) steepest is n^2 = 22500 per pass; with F O(1) each.
            # Use a limited number of random candidate pairs per pass for speed.
            order = rng.permutation(n)
            # Try a subset of pairs to keep per-pass cheap, but steepest needs full scan.
            # Compromise: scan all i, and for each i sample a few k? We'll do full scan but
            # vectorize the delta computation.
            # delta(i,k) = F[i,p[k]] + F[k,p[i]] - F[i,p[i]] - F[k,p[k]]
            # Vectorize over k for each i is O(n^2). Do it in blocks.
            # Precompute col = p
            col = p
            Fi = F  # n x n
            # For each i, delta_i_k = Fi[i, col[k]] + Fi[k, col[i]] - Fi[i, col[i]] - Fi[k, col[k]]
            # = Fi[i, col] + Fi[:, col[i]] - Fi[i, col[i]] - Fi[:, col[k]]
            # where Fi[:, col[k]] is a column vector indexed by k.
            # Let's compute per i:
            #   term1 = Fi[i, col]          # shape (n,) over k
            #   term2 = Fi[:, col[i]]       # shape (n,) over k
            #   term3 = Fi[i, col[i]]       # scalar
            #   term4 = Fi[:, col[k]]       # shape (n,) over k  (this is Fi[:, col] rows)
            # delta = term1 + term2 - term3 - term4
            # term4[k] = Fi[k, col[k]]  -> that's np.einsum('ij,j->i', Fi, col) = current per-facility cost
            cur_per = np.einsum('ij,j->i', Fi, col)  # shape (n,)
            for i in range(n):
                delta = Fi[i, col] + Fi[:, col[i]] - Fi[i, col[i]] - cur_per
                # exclude k == i
                delta[i] = 0
                k = int(np.argmin(delta))
                if delta[k] < best_delta:
                    best_delta = delta[k]
                    best_i = i
                    best_k = k
            if best_delta < 0:
                # apply swap
                p[best_i], p[best_k] = p[best_k], p[best_i]
                improved = True
                # update cur_per for the two facilities
                cur_per[best_i] = Fi[best_i, col[best_i]]
                cur_per[best_k] = Fi[best_k, col[best_k]]
        return p, int(np.einsum('ij,j->', F, p))

    # ---------- initial solutions ----------
    def init_rank():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def init_reverse():
        heavy = np.argsort(a.sum(axis=1), kind="stable")
        central = np.argsort(-b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def init_random():
        return rng.permutation(n).astype(np.int64)

    best_p = None
    best_cost = None

    # Run local search from several starts
    starts = [init_rank(), init_reverse()]
    # add a few random starts
    for _ in range(3):
        starts.append(init_random())

    for p0 in starts:
        if time.time() >= deadline:
            break
        p, c = local_search(p0.copy())
        if best_cost is None or c < best_cost:
            best_cost = c
            best_p = p.copy()

    # ILS: perturb best and re-local-search while time remains
    while time.time() < deadline - 5.0:
        if best_p is None:
            break
        p = best_p.copy()
        # perturb: do a few random swaps
        nswaps = 5
        for _ in range(nswaps):
            i, k = rng.integers(0, n, size=2)
            if i != k:
                p[i], p[k] = p[k], p[i]
        p, c = local_search(p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    if best_p is None:
        return init_rank().tolist()
    return best_p.tolist()
# EVOLVE-BLOCK-END
