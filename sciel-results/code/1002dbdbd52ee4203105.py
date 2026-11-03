"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _delta_2swap(a, b, p, i, j):
    """Compute delta cost for swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    # Direct interaction between i and j
    old_ij = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    new_ij = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    
    # Interactions with all other facilities
    # For facility i: sum over k != i,j of a[i,k]*(b[pj, p[k]] - b[pi, p[k]])
    #                  + a[k,i]*(b[p[k], pj] - b[p[k], pi])
    # For facility j: sum over k != i,j of a[j,k]*(b[pi, p[k]] - b[pj, p[k]])
    #                  + a[k,j]*(b[p[k], pi] - b[p[k], pj])
    
    # Vectorized computation
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = np.where(mask)[0]
    
    if len(others) == 0:
        return new_ij - old_ij
    
    pk = p[others]
    
    delta = new_ij - old_ij
    delta += np.sum(a[i, others] * (b[pj, pk] - b[pi, pk]))
    delta += np.sum(a[others, i] * (b[pk, pj] - b[pk, pi]))
    delta += np.sum(a[j, others] * (b[pi, pk] - b[pj, pk]))
    delta += np.sum(a[others, j] * (b[pk, pi] - b[pk, pj]))
    
    return delta


def _greedy_construct(a, b, order):
    """Greedy construction: place facilities in given order, each to best available location."""
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Precompute row sums for tie-breaking
    for step, fac in enumerate(order):
        if step == 0:
            # First facility: pick location minimizing sum of a[fac, :] * b[:, loc] + a[:, fac] * b[loc, :]
            # Actually just pick the location with smallest total interaction
            scores = np.zeros(n)
            for loc in range(n):
                scores[loc] = np.sum(a[fac, :] * b[loc, :]) + np.sum(a[:, fac] * b[:, loc])
            best_loc = np.argmin(scores)
        else:
            # For subsequent facilities, compute marginal cost for each unused location
            placed_facs = order[:step]
            placed_locs = p[placed_facs]
            
            best_loc = -1
            best_cost = np.inf
            for loc in range(n):
                if used[loc]:
                    continue
                # Cost contribution: interactions with already placed facilities
                cost = 0
                for k in range(step):
                    f = placed_facs[k]
                    l = placed_locs[k]
                    cost += a[fac, f] * b[loc, l] + a[f, fac] * b[l, loc]
                if cost < best_cost:
                    best_cost = cost
                    best_loc = loc
        
        p[fac] = best_loc
        used[best_loc] = True
    
    return p


def _local_search_2opt(a, b, p, max_iter=10000, time_limit=None, start_time=None):
    """Local search with 2-opt swaps."""
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    
    if time_limit is None:
        time_limit = 100
    if start_time is None:
        start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _delta_2swap(a, b, p, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    
    return p, current_cost


def _perturb(p, k=4):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Multiple random restarts with greedy construction + local search
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() - start_time < time_budget:
        # Generate a random ordering for greedy construction
        order = list(range(n))
        rng.shuffle(order)
        
        # Greedy construct
        p = _greedy_construct(a, b, order)
        num_restarts += 1
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        p, cost = _local_search_2opt(a, b, p, time_limit=remaining - 1, start_time=start_time)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # If we have a good solution, do iterated local search
        if best_cost < 50000000 and time.time() - start_time < time_budget - 10:
            # Perturb and local search again
            for _ in range(3):
                if time.time() - start_time > time_budget - 5:
                    break
                p_pert = _perturb(best_p, k=3)
                remaining = time_budget - (time.time() - start_time)
                if remaining < 2:
                    break
                p_ls, cost_ls = _local_search_2opt(a, b, p_pert, time_limit=remaining - 1, start_time=start_time)
                if cost_ls < best_cost:
                    best_cost = cost_ls
                    best_p = p_ls.copy()
    
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n, dtype=np.int64)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
