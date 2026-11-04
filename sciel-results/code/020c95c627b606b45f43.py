"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _swap_delta(a, b, p, i, j):
    """Compute the delta cost for swapping positions i and j in permutation p."""
    x = p[i]
    y = p[j]
    if x == y:
        return 0
    
    delta = 0
    # Terms involving i and j directly
    # a[i,i]: b[x,x] -> b[y,y]
    delta += a[i, i] * (b[y, y] - b[x, x])
    # a[i,j]: b[x,y] -> b[y,x]
    delta += a[i, j] * (b[y, x] - b[x, y])
    # a[j,i]: b[y,x] -> b[x,y]
    delta += a[j, i] * (b[x, y] - b[y, x])
    # a[j,j]: b[y,y] -> b[x,x]
    delta += a[j, j] * (b[x, x] - b[y, y])
    
    # Terms with other indices k
    for k in range(len(p)):
        if k == i or k == j:
            continue
        pk = p[k]
        # a[i,k]: b[x,pk] -> b[y,pk]
        delta += a[i, k] * (b[y, pk] - b[x, pk])
        # a[k,i]: b[pk,x] -> b[pk,y]
        delta += a[k, i] * (b[pk, y] - b[pk, x])
        # a[j,k]: b[y,pk] -> b[x,pk]
        delta += a[j, k] * (b[x, pk] - b[y, pk])
        # a[k,j]: b[pk,y] -> b[pk,x]
        delta += a[k, j] * (b[pk, x] - b[pk, y])
    
    return delta


def _local_search(a, b, p, time_limit):
    """Hill climbing with 2-swap moves until no improvement or time limit."""
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_pair = None
        
        # Evaluate all 2-swap moves
        for i in range(n):
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_pair = (i, j)
        
        if best_pair is not None and best_delta < 0:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            improved = True
    
    return p, current_cost


def _simulated_annealing(a, b, p, start_temp, end_temp, time_limit):
    """Simulated annealing with 2-swap moves."""
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    # Adaptive cooling
    total_time = time_limit
    elapsed = 0
    
    while (time.time() - start_time) < time_limit:
        temp = start_temp * (end_temp / start_temp) ** (elapsed / total_time)
        if temp < end_temp:
            temp = end_temp
        
        # Generate random swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or random.random() < np.exp(-delta / temp):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        elapsed = time.time() - start_time
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    current_cost = _compute_cost(a, b, np.array(p))
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 1: Local search to converge
    time_left = deadline - time.time()
    if time_left > 5:
        p, current_cost = _local_search(a, b, p, time_left - 5)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Phase 2: Simulated annealing with restarts
    time_left = deadline - time.time()
    if time_left > 10:
        # Run SA from current best
        sa_time = min(time_left - 5, 60)
        p_sa, cost_sa = _simulated_annealing(a, b, best_p.copy(), 1000.0, 1.0, sa_time)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
        
        # Local search on SA result
        time_left = deadline - time.time()
        if time_left > 5:
            p_ls, cost_ls = _local_search(a, b, best_p.copy(), time_left - 5)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
        
        # More restarts if time permits
        time_left = deadline - time.time()
        while time_left > 15:
            # Perturb best solution
            p_pert = best_p.copy()
            for _ in range(5):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            sa_time = min(time_left - 10, 30)
            p_sa, cost_sa = _simulated_annealing(a, b, p_pert, 500.0, 1.0, sa_time)
            
            time_left = deadline - time.time()
            if time_left > 5:
                p_ls, cost_ls = _local_search(a, b, p_sa, time_left - 5)
                if cost_ls < cost_sa:
                    cost_sa = cost_ls
                    p_sa = p_ls
            
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
            
            time_left = deadline - time.time()
    
    # Final local search pass
    time_left = deadline - time.time()
    if time_left > 3:
        p_final, cost_final = _local_search(a, b, best_p.copy(), time_left - 2)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p
# EVOLVE-BLOCK-END
