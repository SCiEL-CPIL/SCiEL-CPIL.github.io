"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j
    in the permutation p (i.e., swapping p[i] and p[j]).
    
    p[i] -> p[j], p[j] -> p[i]
    Let u = p[i], v = p[j].
    
    The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    Only terms involving p[i] or p[j] change.
    
    Changed terms:
    - a[i,i]*b[u,u] -> a[i,i]*b[v,v]
    - a[j,j]*b[v,v] -> a[j,j]*b[u,u]
    - a[i,j]*b[u,v] -> a[i,j]*b[v,u]
    - a[j,i]*b[v,u] -> a[j,i]*b[u,v]
    - For k != i,j: a[i,k]*b[u,p[k]] -> a[i,k]*b[v,p[k]]
    - For k != i,j: a[j,k]*b[v,p[k]] -> a[j,k]*b[u,p[k]]
    - For k != i,j: a[k,i]*b[p[k],u] -> a[k,i]*b[p[k],v]
    - For k != i,j: a[k,j]*b[p[k],v] -> a[k,j]*b[p[k],u]
    
    Delta = new - old
    """
    u = p[i]
    v = p[j]
    
    # Terms that swap between i and j
    # a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
    old_ii = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    # a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
    new_ii = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    # Terms with other positions k
    # For k != i,j:
    # old: a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]] + a[k,i]*b[p[k],u] + a[k,j]*b[p[k],v]
    # new: a[i,k]*b[v,p[k]] + a[j,k]*b[u,p[k]] + a[k,i]*b[p[k],v] + a[k,j]*b[p[k],u]
    
    # Vectorized computation for all k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) > 0:
        pk = p[k_indices]
        
        # a[i,k] for all k in k_indices
        a_i_k = a[i, k_indices]
        a_j_k = a[j, k_indices]
        a_k_i = a[k_indices, i]
        a_k_j = a[k_indices, j]
        
        # b[u,pk], b[v,pk]
        b_u_pk = b[u, pk]
        b_v_pk = b[v, pk]
        b_pk_u = b[pk, u]
        b_pk_v = b[pk, v]
        
        old_cross = np.sum(a_i_k * b_u_pk + a_j_k * b_v_pk + a_k_i * b_pk_u + a_k_j * b_pk_v)
        new_cross = np.sum(a_i_k * b_v_pk + a_j_k * b_u_pk + a_k_i * b_pk_v + a_k_j * b_pk_u)
    else:
        old_cross = 0
        new_cross = 0
    
    delta = (new_ii + new_cross) - (old_ii + old_cross)
    return int(delta)


def _two_opt_search(a, b, p, max_iterations=100000, time_limit=None, start_time=None):
    """
    Perform 2-opt local search on permutation p.
    Returns (best_p, best_cost).
    """
    if time_limit is None:
        time_limit = 100
    if start_time is None:
        start_time = time.time()
    
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Check time limit
        if time.time() - start_time > time_limit - 2:
            break
        
        # Try all pairs (i, j) with i < j
        # To speed up, we can sample or use a heuristic order
        # For now, try all pairs but break early if time is running out
        
        for i in range(n):
            if time.time() - start_time > time_limit - 2:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 1:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
    
    return best_p, best_cost


def _random_restart_search(a, b, time_budget=110):
    """
    Perform multiple random restarts with 2-opt local search.
    """
    start_time = time.time()
    n = len(a)
    
    # Precompute row sums for rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    best_p = None
    best_cost = float('inf')
    
    # First, try the rank-matching heuristic
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    cost = _qap_cost(a, b, p)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    # Now do 2-opt from this start
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        p_opt, cost_opt = _two_opt_search(a, b, p, time_limit=remaining_time - 2, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Try random restarts if time allows
    rng = np.random.default_rng(42)
    restart_count = 0
    while time.time() - start_time < time_budget - 10:
        restart_count += 1
        
        # Generate a random permutation
        p_rand = rng.permutation(n)
        
        # Optionally, do a quick greedy improvement or just 2-opt
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        # Use a shorter time limit for each restart
        per_restart_time = min(remaining_time - 2, 15)
        if per_restart_time < 2:
            break
            
        p_opt, cost_opt = _two_opt_search(a, b, p_rand, time_limit=per_restart_time, start_time=time.time())
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    return best_p, best_cost


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: leave 10s margin for safety
    time_budget = 110
    
    best_p, best_cost = _random_restart_search(a, b, time_budget=time_budget)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
