"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _rank_matching_init(a, b):
    """Rank-matching heuristic: heavy flows to short distances."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _2opt_step(a, b, p, i, j):
    """
    Try swapping positions i and j in the permutation p.
    Returns (delta_cost, new_p) where delta_cost is cost_after - cost_before.
    We compute the cost change analytically for speed.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # Cost contribution involving i and j before swap:
    # sum_k a[i,k]*b[pi,p[k]] + sum_k a[k,i]*b[p[k],pi] 
    # + sum_k a[j,k]*b[pj,p[k]] + sum_k a[k,j]*b[p[k],pj]
    # + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # After swap, pi and pj are exchanged.
    
    # More efficient: compute the change directly.
    # Let's just recompute the affected rows/cols.
    
    # Before: terms involving facility i at loc pi and facility j at loc pj
    # The change in cost from swapping pi and pj:
    
    # For all k != i, j:
    #   Before: a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    #   After:  a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    #   Diff: a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
    #         + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # For the i-j pair:
    #   Before: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    #   After:  a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    
    # Let's compute this using vectorized operations for speed.
    
    # Exclude i and j from the main sum
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    k_idx = np.where(mask)[0]
    
    if len(k_idx) == 0:
        # Only i and j matter
        delta = (a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]) - (a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi])
        return delta, p
    
    pk = p[k_idx]
    
    # Terms for facility i
    diff_i = a[i, k_idx] * (b[pj, pk] - b[pi, pk]) + a[k_idx, i] * (b[pk, pj] - b[pk, pi])
    # Terms for facility j
    diff_j = a[j, k_idx] * (b[pi, pk] - b[pj, pk]) + a[k_idx, j] * (b[pk, pi] - b[pk, pj])
    
    main_delta = np.sum(diff_i) + np.sum(diff_j)
    
    # Cross terms between i and j
    cross_delta = (a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]) - (a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi])
    
    delta = main_delta + cross_delta
    return delta, p


def _local_search(a, b, p, max_iter=5000, time_limit=5.0):
    """
    2-opt local search: repeatedly swap pairs to reduce cost.
    Uses random pair selection with some greedy bias.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random positions to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        delta, _ = _2opt_step(a, b, p, i, j)
        
        if delta < 0:
            # Accept the swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost


def _simulated_annealing(a, b, p_init, t0=1000.0, t_end=1.0, cooling=0.999, 
                          max_iter=50000, time_limit=10.0):
    """
    Simulated annealing with 2-opt moves.
    """
    n = len(p)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    t = t0
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random positions
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        delta, _ = _2opt_step(a, b, p, i, j)
        
        if delta < 0 or rng.random() < np.exp(-delta / t):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        t *= cooling
        if t < t_end:
            t = t_end
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with rank-matching heuristic
    p = _rank_matching_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 1: Local search to polish the initial solution
    if time.time() < deadline - 20:
        p_ls, cost_ls = _local_search(a, b, p, max_iter=20000, time_limit=15.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: Multiple SA restarts from perturbed solutions
    rng = np.random.default_rng(99)
    sa_budget = deadline - time.time() - 5.0  # Leave 5s margin
    if sa_budget > 5:
        num_restarts = max(1, int(sa_budget / 10.0))
        per_restart_time = min(10.0, sa_budget / num_restarts)
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution found so far
            p_perturbed = best_p.copy()
            # Randomly swap a few pairs
            num_swaps = max(3, n // 10)
            for _ in range(num_swaps):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
            p_sa, cost_sa = _simulated_annealing(
                a, b, p_perturbed,
                t0=500.0, t_end=1.0, cooling=0.9995,
                max_iter=30000, time_limit=per_restart_time
            )
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
