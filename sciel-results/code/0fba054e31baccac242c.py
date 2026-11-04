"""
QAPLIB wil100 solver: Hungarian init + fixed 2-opt delta + 3-opt cyclic neighborhood.

Key fixes and improvements:
1. Correct swap-delta formula (the previous one double-counted the (i,j)/(j,i)
   terms, producing wrong deltas and corrupting local search).
2. 3-opt cyclic exchange neighborhood (rotate three facilities among three
   locations) to escape 2-opt local optima, with an exact O(n) delta formula.
3. Hungarian (scipy) initialization from rank-matched row/col sums.
4. Iterated Local Search alternating 2-opt and 3-opt, with SA acceptance,
   followed by a final SA polish. Explicit deadline guard throughout.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Correct cost change when swapping positions i and j in permutation p.

    Let x=p[i], y=p[j]. After swap: pos i holds y, pos j holds x.
    Derivation (verified):
      delta = sum_k (a[i,k]-a[j,k])*(b[y,p[k]]-b[x,p[k]])
            + sum_k (a[k,i]-a[k,j])*(b[p[k],y]-b[p[k],x])
            - (a[i,i]-a[j,i])*(b[y,x]-b[x,x])
            - (a[i,j]-a[j,j])*(b[y,y]-b[x,y])
            - (a[i,i]-a[i,j])*(b[x,y]-b[x,x])
            - (a[j,i]-a[j,j])*(b[y,y]-b[y,x])
    """
    x = p[i]
    y = p[j]
    dar = a[i, :] - a[j, :]
    dac = a[:, i] - a[:, j]
    dbr = b[p, y] - b[p, x]
    dbc = b[y, p] - b[x, p]
    s1 = int(np.dot(dar, dbc))
    s2 = int(np.dot(dac, dbr))
    c1 = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
    c2 = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
    c3 = (a[i, i] - a[i, j]) * (b[x, y] - b[x, x])
    c4 = (a[j, i] - a[j, j]) * (b[y, y] - b[y, x])
    return int(s1 + s2 - c1 - c2 - c3 - c4)

def _c3_delta(a, b, p, i, j, k):
    """
    Exact cost change for cyclic exchange:
      pos i <- p[j], pos j <- p[k], pos k <- p[i]
    (x=p[i], y=p[j], z=p[k]; after: pos i=y, pos j=z, pos k=x)

    delta = sum_k a[i,k]*(b[y,p[k]]-b[x,p[k]])
          + sum_k a[j,k]*(b[z,p[k]]-b[y,p[k]])
          + sum_k a[k,i]*(b[p[k],y]-b[p[k],x])
          + sum_k a[k,j]*(b[p[k],z]-b[p[k],y])
          + (a[i,i]*(b[x,y]-b[x,x]) + a[j,i]*(b[x,z]-b[x,y])
           + a[i,j]*(b[y,y]-b[y,x]) + a[j,j]*(b[y,z]-b[y,y])
           + a[i,k]*(b[x,z]-b[x,y]) + a[j,k]*(b[y,x]-b[y,z]))
    """
    x = p[i]
    y = p[j]
    z = p[k]
    s1 = int(np.dot(a[i, :], b[y, p] - b[x, p]))
    s2 = int(np.dot(a[j, :], b[z, p] - b[y, p]))
    s3 = int(np.dot(a[:, i], b[p, y] - b[p, x]))
    s4 = int(np.dot(a[:, j], b[p, z] - b[p, y]))
    c = (a[i, i] * (b[x, y] - b[x, x])
         + a[j, i] * (b[x, z] - b[x, y])
         + a[i, j] * (b[y, y] - b[y, x])
         + a[j, j] * (b[y, z] - b[y, y])
         + a[i, k] * (b[x, z] - b[x, y])
         + a[j, k] * (b[y, x] - b[y, z]))
    return int(s1 + s2 + s3 + s4 + c)

def _hungarian_initial(a, b):
    """Initialize by matching facilities to locations via Hungarian on a
    surrogate cost matrix built from row/col centrality products."""
    n = a.shape[0]
    ar = a.sum(axis=1)
    ac = a.sum(axis=0)
    br = b.sum(axis=1)
    bc = b.sum(axis=0)
    # Surrogate: facility i prefers location l with high (ar[i]+ac[i]) * (br[l]+bc[l])
    # Minimize negative product.
    surr = -np.outer(ar + ac, br + bc)
    row_ind, col_ind = linear_sum_assignment(surr)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _local_search_2(a, b, p, deadline, best_p, best_cost):
    """First-improvement 2-opt local search. Mutates p in place.
    Returns (p, cost, best_p, best_cost)."""
    n = len(p)
    cost = _qap_cost(a, b, p)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    no_improve = 0
    max_no_improve = 400
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        if d < 0:
            p[i], p[j] = p[j], p[i]
            cost += d
            no_improve = 0
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    return p, cost, best_p, best_cost

def _local_search_3(a, b, p, deadline, best_p, best_cost):
    """First-improvement 3-opt cyclic local search. Mutates p in place."""
    n = len(p)
    cost = _qap_cost(a, b, p)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    no_improve = 0
    max_no_improve = 600
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        k = np.random.randint(n)
        if i == j or i == k or j == k:
            continue
        d = _c3_delta(a, b, p, i, j, k)
        if d < 0:
            # apply cyclic: pos i<-y, pos j<-z, pos k<-x
            x, y, z = p[i], p[j], p[k]
            p[i], p[j], p[k] = y, z, x
            cost += d
            no_improve = 0
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    return p, cost, best_p, best_cost

def _perturb(p, k):
    """Apply k random swaps, return a copy."""
    n = len(p)
    q = p.copy()
    for _ in range(k):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            q[i], q[j] = q[j], q[i]
    return q

def _sa_polish(a, b, p, deadline, best_p, best_cost):
    """Simulated annealing polish using 2-opt moves. Mutates p in place."""
    n = len(p)
    cost = _qap_cost(a, b, p)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    T_start = 500.0
    T_end = 0.5
    while time.time() < deadline:
        frac = min(1.0, (time.time() - _sa_polish.t0) / max(_sa_polish.t1, 1.0))
        T = T_start * (T_end / T_start) ** frac
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or np.random.rand() < np.exp(-d / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cost += d
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    return p, cost, best_p, best_cost

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    np.random.seed(42)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s

    best_p = None
    best_cost = np.inf

    # Phase 1: Hungarian init + local search
    p = _hungarian_initial(a, b)
    p, cost, best_p, best_cost = _local_search_2(a, b, p, min(t0 + 8.0, deadline), best_p, best_cost)

    # Phase 2: Random restarts with short local search
    for r in range(20):
        if time.time() > deadline - 35.0:
            break
        p_rand = np.random.permutation(n).astype(np.int64)
        ls_end = min(t0 + 8.0, deadline - 30.0)
        p, cost, best_p, best_cost = _local_search_2(a, b, p_rand, ls_end, best_p, best_cost)

    # Phase 3: Iterated Local Search alternating 2-opt and 3-opt
    if time.time() < deadline - 15.0:
        p_ils = best_p.copy()
        cost_ils = _qap_cost(a, b, p_ils)
        pert_k = 3
        use_3 = False
        while time.time() < deadline - 12.0:
            p_pert = _perturb(p_ils, pert_k)
            ls_end = min(time.time() + 4.0, deadline - 10.0)
            if use_3:
                p_ls, cost_ls, best_p, best_cost = _local_search_3(a, b, p_pert, ls_end, best_p, best_cost)
            else:
                p_ls, cost_ls, best_p, best_cost = _local_search_2(a, b, p_pert, ls_end, best_p, best_cost)
            use_3 = not use_3
            if cost_ls < cost_ils:
                p_ils = p_ls
                cost_ils = cost_ls
                pert_k = max(2, pert_k - 1)
            else:
                delta = cost_ls - cost_ils
                T = 80.0
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    p_ils = p_ls
                    cost_ils = cost_ls
                pert_k = min(8, pert_k + 1)
            if cost_ils < best_cost:
                best_cost = cost_ils
                best_p = p_ils.copy()

    # Phase 4: Final SA polish
    if time.time() < deadline - 2.0:
        _sa_polish.t0 = time.time()
        _sa_polish.t1 = deadline - time.time()
        p_final, cost_final, best_p, best_cost = _sa_polish(
            a, b, best_p.copy(), deadline, best_p, best_cost)

    return best_p.tolist()

# EVOLVE-BLOCK-END
