"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _compute_delta(a, D, p, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    D = b[p[:,p]] is the current distance matrix in permutation order.
    """
    fi = p[i]
    fj = p[j]
    
    # Use vectorized numpy operations for O(n) delta computation
    # The delta formula for 2-swap in QAP:
    # delta = sum_k [a[fi,k]*(D[j,k]-D[i,k]) + a[k,fi]*(D[k,j]-D[k,i])
    #                   + a[fj,k]*(D[i,k]-D[j,k]) + a[k,fj]*(D[k,i]-D[k,j])]
    #       - a[fi,fj]*(D[j,i]-D[i,j]) - a[fj,fi]*(D[i,j]-D[j,i])
    
    # Vectorized computation
    diff_row = D[j, :] - D[i, :]  # shape (n,)
    diff_col = D[:, j] - D[:, i]  # shape (n,)
    
    # Sum over k of a[fi,k]*diff_row[k]
    term1 = np.dot(a[fi, :], diff_row)
    
    # Sum over k of a[k,fi]*diff_col[k]
    term2 = np.dot(a[:, fi], diff_col)
    
    # Sum over k of a[fj,k]*(-diff_row[k])
    term3 = -np.dot(a[fj, :], diff_row)
    
    # Sum over k of a[k,fj]*(-diff_col[k])
    term4 = -np.dot(a[:, fj], diff_col)
    
    # Subtract double-counted cross terms
    cross = a[fi, fj] * (D[j, i] - D[i, j]) + a[fj, fi] * (D[i, j] - D[j, i])
    
    return term1 + term2 + term3 + term4 - cross

def _local_search(a, b, p, time_limit, start_time):
    """
    Efficient local search with 2-opt swaps.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _compute_delta(a, D, p, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D: swap rows and columns i and j
                    D[i, :], D[j, :] = D[j, :].copy(), D[i, :].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    # Continue scanning (first-improvement strategy)
    
    return p, current_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    rng = random.Random(42)
    
    # Multiple random restarts with local search
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Random initial permutation
        p_init = np.arange(n, dtype=np.int64)
        rng.shuffle(p_init.tolist())
        p_init = np.array(p_init, dtype=np.int64)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, remaining - 2, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Perturb best solution
        k_perturb = random.randint(3, 10)
        p_pert = best_p.copy()
        for _ in range(k_perturb):
            i, j = random.sample(range(n), 2)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
