# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _delta_swap_cost(p, i, j, a, b):
    """Compute delta cost when swapping positions i and j in permutation p."""
    fi = p[i]
    fj = p[j]
    if fi == fj:
        return 0
    
    n = len(p)
    delta = 0
    delta += a[i, i] * (b[fj, fj] - b[fi, fi])
    delta += a[i, j] * (b[fj, fi] - b[fi, fj])
    delta += a[j, i] * (b[fi, fj] - b[fj, fi])
    delta += a[j, j] * (b[fi, fi] - b[fj, fj])
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    pv = p[mask]
    
    delta += np.sum((a[i, mask] - a[j, mask]) * (b[fj, pv] - b[fi, pv]))
    delta += np.sum((a[mask, i] - a[mask, j]) * (b[pv, fj] - b[pv, fi]))
    
    return int(delta)


def _local_search(p, a, b, max_time):
    n = len(p)
    current_cost = _calc_cost(a, b, p)
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for i in range(n):
            if time.time() >= max_time:
                break
            for j in range(i + 1, n):
                if time.time() >= max_time:
                    break
                delta = _delta_swap_cost(p, i, j, a, b)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    return p, current_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0
    
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    rng = np.random.default_rng(42)
    
    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = _calc_cost(a, b, p)
    
    # Local search
    if time.time() < deadline - 20.0:
        p, cost = _local_search(p, a, b, deadline - 20.0)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Simulated annealing
    if time.time() < deadline - 15.0:
        current_p = best_p.copy()
        current_cost = best_cost
        T = 5000.0
        cooling = 0.9995
        
        while time.time() < deadline - 10.0 and T > 1.0:
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            delta = _delta_swap_cost(current_p, i, j, a, b)
            if delta < 0 or (T > 1.0 and rng.random() < np.exp(-delta / T)):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            T *= cooling
            
            # Periodic local search
            if T < 500.0 and time.time() < deadline - 15.0:
                current_p, current_cost = _local_search(current_p, a, b, deadline - 15.0)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
    
    # Final local search
    if time.time() < deadline - 2.0:
        best_p, best_cost = _local_search(best_p, a, b, deadline - 1.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
