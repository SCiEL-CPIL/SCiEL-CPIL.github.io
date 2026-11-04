"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """Compute cost change from swapping facilities i and j in assignment p."""
    pi, pj = p[i], p[j]
    # Delta = sum_k [a[i,k]*(b[pj,k]-b[pi,k]) + a[j,k]*(b[pi,k]-b[pj,k]) + a[k,i]*(b[k,pj]-b[k,pi]) + a[k,j]*(b[k,pi]-b[k,pj])]
    # = (a[i,:] - a[j,:]) @ (b[pj,:] - b[pi,:]) + (a[:,i] - a[:,j]) @ (b[:,pj] - b[:,pi])
    row_diff = a[i, :] - a[j, :]
    col_diff = a[:, i] - a[:, j]
    loc_diff = b[pj, :] - b[pi, :]
    loc_diff_col = b[:, pj] - b[:, pi]
    delta = np.dot(row_diff, loc_diff) + np.dot(col_diff, loc_diff_col)
    return int(delta)


def _local_search_2opt(a, b, p, time_limit, deadline):
    """2-opt local search: repeatedly swap pairs that reduce cost until no improvement or time limit."""
    n = len(p)
    p = np.array(p, dtype=np.int64)
    current_cost = _compute_cost(a, b, p)
    
    # Precompute row and column sums for efficiency
    # We'll iterate through all pairs and find the best improving swap each pass
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Check all pairs
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0:
            # Apply the best swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
    
    return p.tolist(), current_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Multiple restarts with different initial solutions
    best_p = None
    best_cost = np.iinfo(np.int64).max
    
    # Strategy: rank-matching + random perturbations, then local search
    # Rank-matching initial solution
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    # Try multiple initial solutions
    rng = np.random.default_rng(42)
    
    # First: pure rank-matching
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_list, cost = _local_search_2opt(a, b, p_init.tolist(), n, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p_list
    
    if time.time() >= deadline:
        return best_p
    
    # Second: random initial solution
    p_rand = rng.permutation(n)
    p_list, cost = _local_search_2opt(a, b, p_rand.tolist(), n, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p_list
    
    if time.time() >= deadline:
        return best_p
    
    # Third: rank-matching with random perturbation
    p_pert = p_init.copy()
    # Swap a few random pairs
    for _ in range(10):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
    p_list, cost = _local_search_2opt(a, b, p_pert.tolist(), n, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p_list
    
    if time.time() >= deadline:
        return best_p
    
    # Fourth: another random
    p_rand2 = rng.permutation(n)
    p_list, cost = _local_search_2opt(a, b, p_rand2.tolist(), n, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p_list
    
    if time.time() >= deadline:
        return best_p
    
    # Fifth: rank-matching with different perturbation
    p_pert2 = p_init.copy()
    for _ in range(20):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p_pert2[i], p_pert2[j] = p_pert2[j], p_pert2[i]
    p_list, cost = _local_search_2opt(a, b, p_pert2.tolist(), n, deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p_list
    
    return best_p
# EVOLVE-BLOCK-END
