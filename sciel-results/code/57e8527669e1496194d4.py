"""QAP solver for tai256c: multi-start local search with 2-opt/3-opt moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _delta_2opt(a, b, p, i, j):
    """Compute cost delta of swapping positions i and j in permutation p."""
    pi, pj = p[i], p[j]
    # For each facility k, the change in contribution involving i or j
    # Old: a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    #      + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New: a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    #      + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    
    # Vectorized computation over all k
    # Let's compute more carefully:
    # The total cost involves all pairs (i,j). When we swap p[i] and p[j], 
    # only terms involving i or j change.
    
    # Terms with i (as first index) and k (as second), k != j:
    # Old: sum_{k!=j} a[i,k]*b[pi,p[k]]  New: sum_{k!=j} a[i,k]*b[pj,p[k]]
    # Terms with i (as second index) and k (as first), k != j:
    # Old: sum_{k!=j} a[k,i]*b[p[k],pi]  New: sum_{k!=j} a[k,i]*b[p[k],pj]
    # Terms with j (as first index) and k (as second), k != i:
    # Old: sum_{k!=i} a[j,k]*b[pj,p[k]]  New: sum_{k!=i} a[j,k]*b[pi,p[k]]
    # Terms with j (as second index) and k (as first), k != i:
    # Old: sum_{k!=i} a[k,j]*b[p[k],pj]  New: sum_{k!=i} a[k,j]*b[p[k],pi]
    # Terms (i,j) and (j,i):
    # Old: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    
    n = len(p)
    # Build index array excluding i and j
    idx = np.arange(n)
    mask = (idx != i) & (idx != j)
    k_idx = idx[mask]
    
    # For terms involving i as first index, k as second (k != j)
    old_ik = np.sum(a[i, k_idx] * b[pi, p[k_idx]])
    new_ik = np.sum(a[i, k_idx] * b[pj, p[k_idx]])
    
    # For terms involving k as first, i as second (k != j)
    old_ki = np.sum(a[k_idx, i] * b[p[k_idx], pi])
    new_ki = np.sum(a[k_idx, i] * b[p[k_idx], pj])
    
    # For terms involving j as first, k as second (k != i)
    old_jk = np.sum(a[j, k_idx] * b[pj, p[k_idx]])
    new_jk = np.sum(a[j, k_idx] * b[pi, p[k_idx]])
    
    # For terms involving k as first, j as second (k != i)
    old_kj = np.sum(a[k_idx, j] * b[p[k_idx], pj])
    new_kj = np.sum(a[k_idx, j] * b[p[k_idx], pi])
    
    # Cross terms (i,j) and (j,i)
    old_ij = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    new_ij = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    
    delta = (new_ik + new_ki + new_jk + new_kj + new_ij) - (old_ik + old_ki + old_jk + old_kj + old_ij)
    return delta

def _greedy_initial(a, b, n, rng):
    """Greedy construction: place facilities one by one, choosing best location."""
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    remaining_locs = set(range(n))
    
    # Order facilities by decreasing total flow
    order = np.argsort(-a.sum(axis=1), kind='stable')
    
    for idx, fac in enumerate(order):
        if idx == n - 1:
            p[fac] = list(remaining_locs)[0]
            break
        # For each unplaced location, compute marginal cost
        best_loc = -1
        best_cost = np.inf
        for loc in remaining_locs:
            # Cost of placing fac at loc given already placed facilities
            c = 0
            for k in range(idx):
                placed_fac = order[k]
                c += a[fac, placed_fac] * b[loc, p[placed_fac]]
                c += a[placed_fac, fac] * b[p[placed_fac], loc]
            if c < best_cost:
                best_cost = c
                best_loc = loc
        p[fac] = best_loc
        remaining_locs.remove(best_loc)
    
    return p.tolist()

def _rank_initial(a, b, n):
    """Rank-matching initialization."""
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.tolist()

def _local_search(a, b, p, deadline, max_passes=50):
    """2-opt local search with time budget."""
    p = np.array(p, dtype=np.int64)
    n = len(p)
    best_cost = _qap_cost(a, b, p)
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        # Random order of pairs to avoid cycling
        order = np.random.permutation(n * (n-1) // 2)
        for idx in order:
            if time.time() > deadline:
                break
            i = idx // (n - 1)
            j = idx % (n - 1)
            if j >= i:
                j += 1
            if i == j:
                continue
            delta = _delta_2opt(a, b, p, i, j)
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                best_cost += delta
                improved = True
        # After a full pass, if no improvement, try a second pass with different order
        if not improved:
            break
    
    return p.tolist(), best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 130  # Safety margin: 20s before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # 1. Rank-matching
    p_rank = _rank_initial(a, b, n)
    initial_solutions.append(p_rank)
    
    # 2. Greedy (may be slow for n=256, limit time)
    if time.time() < deadline - 60:
        p_greedy = _greedy_initial(a, b, n, rng)
        initial_solutions.append(p_greedy)
    
    # 3. Random perturbations of rank solution
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        p_pert = np.array(p_rank, dtype=np.int64)
        # Swap random pairs
        for _ in range(10):
            i, j = rng.integers(0, n, 2)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        initial_solutions.append(p_pert.tolist())
    
    best_p = None
    best_cost = np.inf
    
    for sol in initial_solutions:
        if time.time() > deadline:
            break
        p_opt, cost = _local_search(a, b, sol, deadline)
        if cost < best_cost:
            best_cost = cost
            best_p = p_opt
    
    if best_p is None:
        best_p = p_rank
    
    return best_p
# EVOLVE-BLOCK-END
