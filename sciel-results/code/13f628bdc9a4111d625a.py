"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    # p[i] is the location assigned to facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This can be computed as: for each i, sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :]) 
    # More efficiently: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's use the fact that this equals trace-like computation
    # Actually: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each i, a[i,:] dot b[p[i],:]
    # Vectorized: sum over i of a[i,:] @ b[p[i],:]
    # a[i,:] is shape (n,), b[p[i],:] is shape (n,)
    # So for each i, it's a dot product
    # We can do: np.sum(a * b[np.arange(n)[:, None], p][np.arange(n)[:, None], :])
    # Wait, let me think more carefully.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i [ sum_j a[i,j] * b[p[i], p[j]] ]
    # = sum_i [ a[i,:] @ b[p[i],:] ]
    # For each i, a[i,:] is a vector of length n, b[p[i],:] is a vector of length n
    # So a[i,:] @ b[p[i],:] is a scalar
    # We can vectorize: 
    # Let rows_b = b[p, :]  # shape (n, n), row i is b[p[i], :]
    # Then cost = sum_i sum_j a[i,j] * rows_b[i,j]
    # = np.sum(a * rows_b)
    n = len(p)
    rows_b = b[p, :]  # shape (n, n)
    return int(np.sum(a * rows_b))


def _local_search_2opt(a, b, p, max_time=100):
    """Perform 2-opt local search on the QAP solution."""
    n = len(p)
    p = np.array(p, dtype=np.int64)
    best_cost = _compute_cost(a, b, p)
    
    # Precompute for efficiency
    start_time = time.time()
    deadline = start_time + max_time
    
    # For 2-opt, we swap p[i] and p[j]
    # The cost change can be computed incrementally
    # Let's use a simpler approach: try all pairs but with early termination
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Swap p[i] and p[j]
                old_i, old_j = p[i], p[j]
                p[i], p[j] = old_j, old_i
                new_cost = _compute_cost(a, b, p)
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    # Revert
                    p[i], p[j] = old_i, old_j
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return p, best_cost


def _greedy_construct(a, b):
    """Construct a solution using greedy assignment."""
    n = len(a)
    # Use the rank-matching heuristic as a starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _simulated_annealing(a, b, p, max_time=80):
    """Perform simulated annealing on the QAP solution."""
    n = len(p)
    p = np.array(p, dtype=np.int64)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    current_cost = best_cost
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Temperature schedule
    T = 10000.0
    T_min = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        T *= cooling_rate
        if T < T_min:
            break
        
        # Random 2-opt move
        i, j = np.random.choice(n, 2, replace=False)
        old_i, old_j = p[i], p[j]
        p[i], p[j] = old_j, old_i
        new_cost = _compute_cost(a, b, p)
        
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
        else:
            # Revert
            p[i], p[j] = old_i, old_j
    
    return best_p, best_cost


def solve(a, b):
    """Solve the QAP instance."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n = len(a)
    
    # Step 1: Greedy construction
    p = _greedy_construct(a, b)
    cost = _compute_cost(a, b, p)
    
    # Step 2: Local search
    if time.time() < deadline - 30:
        p, cost = _local_search_2opt(a, b, p, max_time=min(30, deadline - time.time() - 20))
    
    # Step 3: Simulated annealing with multiple restarts
    best_p = p.copy()
    best_cost = cost
    
    while time.time() < deadline - 5:
        # Try SA from current best
        p_sa, cost_sa = _simulated_annealing(a, b, best_p, max_time=min(20, deadline - time.time() - 5))
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
