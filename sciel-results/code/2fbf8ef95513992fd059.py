"""QAP solver for tho150 with efficient 2-opt local search and SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute QAP cost: sum a[i,j] * b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute cost delta for swapping positions i and j in permutation p.
    
    O(n) computation without array copies.
    """
    n = len(p)
    u = p[i]
    v = p[j]
    if u == v:
        return 0
    
    # For k != i, j:
    # delta_k = (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    #         + (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    
    # Vectorized over k != i, j
    # b[v, p] - b[u, p] = b[v, p] - b[u, p]  (row v minus row u of b, columns indexed by p)
    # b[p, v] - b[p, u] = b[p, v] - b[p, u]  (col v minus col u of b, rows indexed by p)
    
    diff_row = b[v, p] - b[u, p]  # shape (n,), entry k = b[v,p[k]] - b[u,p[k]]
    diff_col = b[p, v] - b[p, u]  # shape (n,), entry k = b[p[k],v] - b[p[k],u]
    
    # (a[i,:] - a[j,:]) * diff_row, summed over k != i,j
    a_diff_row = a[i, :] - a[j, :]  # shape (n,)
    a_diff_col = a[:, i] - a[:, j]  # shape (n,)
    
    # Sum over all k, then subtract k=i and k=j contributions
    delta = float(np.dot(a_diff_row, diff_row) + np.dot(a_diff_col, diff_col))
    
    # Subtract k=i contribution:
    # (a[i,i] - a[j,i]) * (b[v, p[i]] - b[u, p[i]]) + (a[i,i] - a[i,j]) * (b[p[i], v] - b[p[i], u])
    # p[i] = u, so:
    # = (a[i,i] - a[j,i]) * (b[v,u] - b[u,u]) + (a[i,i] - a[i,j]) * (b[u,v] - b[u,u])
    delta -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
    delta -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
    
    # Subtract k=j contribution:
    # (a[i,j] - a[j,j]) * (b[v, p[j]] - b[u, p[j]]) + (a[j,i] - a[j,j]) * (b[p[j], v] - b[p[j], u])
    # p[j] = v, so:
    # = (a[i,j] - a[j,j]) * (b[v,v] - b[u,v]) + (a[j,i] - a[j,j]) * (b[v,v] - b[v,u])
    delta -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
    delta -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
    
    return delta


def _two_opt(a, b, p, time_limit, start_time, max_passes=100):
    """2-opt local search: repeatedly find best improving swap until no improvement.
    
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    cost = _compute_cost(a, b, p)
    
    passes = 0
    while passes < max_passes:
        if time.time() > start_time + time_limit - 2.0:
            break
        
        best_delta = 0
        best_i, best_j = -1, -1
        passes += 1
        
        for i in range(n):
            if time.time() > start_time + time_limit - 2.0:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_delta - 1e-9:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_i < 0:
            break  # No improvement found
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(round(best_delta))
    
    return p, cost


def _simulated_annealing(a, b, p, cost, time_limit, start_time):
    """Simulated annealing with swap moves.
    
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    
    # Initial temperature based on cost magnitude
    T = max(1.0, cost * 5e-7)
    T_min = 0.1
    alpha = 0.9997
    
    rng = random.Random(42)
    
    while T > T_min:
        if time.time() > start_time + time_limit - 3.0:
            break
        
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        
        d = _swap_delta(a, b, p, i, j)
        
        if d < 0 or rng.random() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            cost += int(round(d))
        
        T *= alpha
    
    return p, cost


def _initial_solution(a, b, seed=0):
    """Generate initial solution using rank matching with randomization."""
    n = len(a)
    rng = np.random.RandomState(seed)
    
    # Rank matching: sort facilities by total flow, locations by total distance
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Add small random perturbation for diversity
    a_perturbed = a_row_sums + rng.randn(n) * 0.01 * np.std(a_row_sums)
    b_perturbed = b_row_sums + rng.randn(n) * 0.01 * np.std(b_row_sums)
    
    facility_order = np.argsort(a_perturbed, kind='stable')
    location_order = np.argsort(b_perturbed, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    return p


def _reduced_cost_assignment(a, b, p):
    """Improve solution using reduced-cost linear assignment.
    
    Compute the reduced cost matrix and solve the assignment problem.
    """
    n = len(a)
    
    # For facility i assigned to location j, the marginal contribution is:
    # sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j]
    # (the last term avoids double-counting the self-interaction)
    
    # Compute efficiently:
    # b_j_p[k] = b[j, p[k]] for all j,k -> this is b[:, p] (n x n matrix)
    b_col_p = b[:, p]  # (n, n): entry (j, k) = b[j, p[k]]
    b_p_col = b[p, :]  # (n, n): entry (k, j) = b[p[k], j]
    
    # cost_matrix[i, j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j]
    # = a[i,:] @ b_col_p[j,:] + a[:,i] @ b_p_col[:,j] - a[i,i]*b[j,j]
    
    # First term: (a @ b_col_p.T)[i, j] = sum_k a[i,k] * b_col_p[j,k] = sum_k a[i,k]*b[j,p[k]]
    term1 = a @ b_col_p.T  # (n, n)
    
    # Second term: (a.T @ b_p_col)[i, j] = sum_k a[k,i] * b_p_col[k,j] = sum_k a[k,i]*b[p[k],j]
    term2 = a.T @ b_p_col  # (n, n)
    
    # Third term: -a[i,i]*b[j,j]
    diag_a = np.diag(a)  # (n,)
    diag_b = np.diag(b)  # (n,)
    term3 = -np.outer(diag_a, diag_b)  # (n, n)
    
    cost_matrix = term1 + term2 + term3
    
    # Solve assignment problem
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    return p_new


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_p = None
    best_cost = float('inf')
    
    # Phase 1: Generate initial solutions and do local search
    n_restarts = 5
    time_per_restart = (deadline - start_time - 10.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial solution
        p = _initial_solution(a, b, seed=restart * 12345 + 7)
        cost = _compute_cost(a, b, p)
        
        # 2-opt local search
        remaining = min(time_per_restart - 0.5, deadline - start_time - 5.0)
        if remaining > 1.0:
            p, cost = _two_opt(a, b, p, remaining, start_time, max_passes=50)
        
        # Track best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Phase 2: SA from best solution
    if time.time() < deadline - 15.0 and best_p is not None:
        sa_time = min(30.0, deadline - start_time - 15.0)
        if sa_time > 5.0:
            p, cost = _simulated_annealing(a, b, best_p, best_cost, sa_time, start_time)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    # Phase 3: Reduced-cost assignment refinement
    if time.time() < deadline - 10.0 and best_p is not None:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            p_new = _reduced_cost_assignment(a, b, best_p)
            cost_new = _compute_cost(a, b, p_new)
            if cost_new < best_cost:
                best_cost = cost_new
                best_p = p_new.copy()
                # Quick local search after assignment
                remaining = deadline - start_time - 5.0
                if remaining > 2.0:
                    best_p, best_cost = _two_opt(a, b, best_p, remaining, start_time, max_passes=20)
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 3.0 and best_p is not None:
        remaining = deadline - start_time - 2.0
        if remaining > 1.0:
            best_p, best_cost = _two_opt(a, b, best_p, remaining, start_time, max_passes=100)
    
    # Final verification
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n, dtype=np.int64)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
