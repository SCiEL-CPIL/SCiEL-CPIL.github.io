"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_build(a, b, rng):
    """Construct a permutation by greedy assignment using combined row+col sums."""
    n = len(a)
    # Combined weight for facilities: row sum + col sum of a
    fa = a.sum(axis=1) + a.sum(axis=0)
    # Combined weight for locations: row sum + col sum of b
    fb = b.sum(axis=1) + b.sum(axis=0)

    # Sort facilities by descending combined weight
    order_f = np.argsort(-fa, kind='stable')
    # Sort locations by ascending combined weight (central locations first)
    order_l = np.argsort(fb, kind='stable')

    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)

    # Greedy: for each facility in order, pick best unused location
    # Cost contribution of placing facility i at location j:
    #   sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
    # We approximate by precomputed arrays.

    # Precompute: for each location j, b[j,:] and b[:,j]
    # As we assign, we build up the running cost.

    # For efficiency, track:
    #   row_contrib[j] = sum over assigned facilities i of a[i, *] dot b[j, p[*]]... 
    # This is getting complex. Let's use a simpler greedy:
    # Assign facility i to location j minimizing a[i,:].dot(b[j, p_assigned]) + a[:,i].dot(b[p_assigned, j])
    # where p_assigned are already assigned locations.

    # Actually, let's do a simpler but effective greedy:
    # Use the fact that cost = sum_{i,j} a[i,j] b[p[i],p[j]]
    # = sum_i sum_j a[i,j] b[p[i],p[j]]
    # For a new assignment of facility i to location j, the marginal cost is:
    #   sum_{k already assigned} a[i,k]*b[j,p[k]] + a[k,i]*b[p[k],j]
    # Plus the "self" term a[i,i]*b[j,j] (usually 0 for QAP)

    # Let's implement a proper greedy:
    p = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)  # which facilities are assigned
    loc_used = np.zeros(n, dtype=bool)

    # Precompute a and b for fast access
    # We'll iterate facilities in order of decreasing combined weight
    for idx in order_f:
        best_loc = -1
        best_cost = np.inf
        for j in range(n):
            if loc_used[j]:
                continue
            # Compute marginal cost of assigning facility idx to location j
            # = sum_{k: assigned} a[idx,k]*b[j,p[k]] + a[k,idx]*b[p[k],j]
            # + a[idx,idx]*b[j,j] (diagonal, often 0)
            c = 0.0
            for k in range(n):
                if assigned[k]:
                    pk = p[k]
                    c += a[idx, k] * b[j, pk]
                    c += a[k, idx] * b[pk, j]
            if c < best_cost:
                best_cost = c
                best_loc = j
        p[idx] = best_loc
        assigned[idx] = True
        loc_used[best_loc] = True

    return p


def _hill_climb_2opt(a, b, p, time_limit=5.0):
    """2-opt local search: swap pairs of locations in the permutation."""
    n = len(a)
    best_p = p.copy()
    best_cost = _cost(a, b, best_p)
    start = time.time()

    # Precompute for faster delta computation
    # For swapping positions i and j in p:
    # delta = sum_k a[i,k]*(b[p[i],p[j]] - b[p[i],p[k]]) + a[k,i]*(b[p[j],p[i]] - b[p[k],p[i]])
    #       + sum_k a[j,k]*(b[p[j],p[i]] - b[p[j],p[k]]) + a[k,j]*(b[p[i],p[j]] - b[p[k],p[j]])
    #       - a[i,j]*(b[p[i],p[j]] - b[p[i],p[i]]) - a[j,i]*(b[p[j],p[i]] - b[p[j],p[j]])
    # This is complex. Let's use a simpler delta formula.

    # When we swap p[i] and p[j], the change in cost is:
    # For all k != i,j:
    #   old: a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
    #   new: a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    #   diff: a[i,k]*(b[p[j],p[k]]-b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]]-b[p[k],p[i]])
    #         + a[j,k]*(b[p[i],p[k]]-b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]]-b[p[k],p[j]])
    # For k=i and k=j:
    #   old: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
    #   new: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #   diff: a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])

    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start > time_limit:
                break
            pi = best_p[i]
            for j in range(i + 1, n):
                if time.time() - start > time_limit:
                    break
                pj = best_p[j]
                if pi == pj:
                    continue
                # Compute delta
                delta = 0.0
                # k = i and j terms
                delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                # k != i, j terms
                # We can vectorize this
                k_mask = np.ones(n, dtype=bool)
                k_mask[i] = False
                k_mask[j] = False
                ks = np.where(k_mask)[0]
                if len(ks) > 0:
                    # a[i,ks] * (b[pj, best_p[ks]] - b[pi, best_p[ks]])
                    delta += np.sum(a[i, ks] * (b[pj, best_p[ks]] - b[pi, best_p[ks]]))
                    # a[ks,i] * (b[best_p[ks],pj] - b[best_p[ks],pi])
                    delta += np.sum(a[ks, i] * (b[best_p[ks], pj] - b[best_p[ks], pi]))
                    # a[j,ks] * (b[pi, best_p[ks]] - b[pj, best_p[ks]])
                    delta += np.sum(a[j, ks] * (b[pi, best_p[ks]] - b[pj, best_p[ks]]))
                    # a[ks,j] * (b[best_p[ks],pi] - b[best_p[ks],pj])
                    delta += np.sum(a[ks, j] * (b[best_p[ks], pi] - b[best_p[ks], pj]))

                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += int(delta)
                    improved = True

    return best_p, best_cost


def _simulated_annealing(a, b, p, time_limit=10.0, rng=None):
    """Simulated annealing with 2-opt moves."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(a)
    cur_p = p.copy()
    cur_cost = _cost(a, b, cur_p)
    best_p = cur_p.copy()
    best_cost = cur_cost

    start = time.time()
    T = 1000.0
    cooling = 0.999

    while time.time() - start < time_limit:
        # Random 2-opt move
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        pi, pj = cur_p[i], cur_p[j]
        if pi == pj:
            continue

        # Compute delta
        delta = 0.0
        delta += a[i, j] * (b[pj, pi] - b[pi, pj])
        delta += a[j, i] * (b[pi, pj] - b[pj, pi])
        k_mask = np.ones(n, dtype=bool)
        k_mask[i] = False
        k_mask[j] = False
        ks = np.where(k_mask)[0]
        if len(ks) > 0:
            delta += np.sum(a[i, ks] * (b[pj, cur_p[ks]] - b[pi, cur_p[ks]]))
            delta += np.sum(a[ks, i] * (b[cur_p[ks], pj] - b[cur_p[ks], pi]))
            delta += np.sum(a[j, ks] * (b[pi, cur_p[ks]] - b[pj, cur_p[ks]]))
            delta += np.sum(a[ks, j] * (b[cur_p[ks], pi] - b[cur_p[ks], pj]))

        new_cost = cur_cost + int(delta)
        if new_cost < cur_cost or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_p[i], cur_p[j] = cur_p[j], cur_p[i]
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = cur_p.copy()
        T *= cooling

    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Phase 1: Greedy construction
    p_greedy = _greedy_build(a, b, np.random.default_rng(42))
    cost_greedy = _cost(a, b, p_greedy)

    best_p = p_greedy.copy()
    best_cost = cost_greedy

    # Phase 2: Hill climbing from greedy
    if time.time() < deadline - 20:
        p_hc, cost_hc = _hill_climb_2opt(a, b, best_p, time_limit=min(15, deadline - time.time() - 15))
        if cost_hc < best_cost:
            best_p = p_hc
            best_cost = cost_hc

    # Phase 3: Simulated annealing
    remaining = deadline - time.time() - 5
    if remaining > 5:
        rng = np.random.default_rng(123)
        p_sa, cost_sa = _simulated_annealing(a, b, best_p, time_limit=remaining, rng=rng)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    # Phase 4: Final hill climb polish
    remaining = deadline - time.time() - 2
    if remaining > 2:
        p_final, cost_final = _hill_climb_2opt(a, b, best_p, time_limit=remaining)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
