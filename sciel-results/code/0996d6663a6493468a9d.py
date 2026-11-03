"""
Quantum-Inspired Annealing for QAPLIB wil100.
Strategy:
1. Use gradient-guided continuous relaxation to initialize a good permutation.
2. Apply efficient best-improvement 2-opt local search.
3. Use simulated annealing with geometric cooling for final polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _best_improvement_2opt(a, b, p, time_limit, start_time):
    """
    Best-improvement 2-opt local search.
    Precomputes all swap deltas each iteration for efficiency.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        # Compute all swap deltas
        deltas = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                deltas[i, j] = _swap_delta(a, b, p, i, j)
                deltas[j, i] = deltas[i, j]
        
        # Find best improvement
        min_idx = np.unravel_index(np.argmin(deltas), deltas.shape)
        min_delta = deltas[min_idx]
        
        if min_delta >= 0:
            break  # Local optimum
        
        i, j = min_idx
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _hungarian_initial(a, b):
    """
    Initialize permutation using Hungarian algorithm on a surrogate cost matrix.
    The surrogate combines row/col sums of A and B to guide the initial assignment.
    """
    n = len(a)
    
    # Compute facility flow centrality (row + col sums of A)
    flow_centrality = a.sum(axis=1) + a.sum(axis=0)
    
    # Compute location distance centrality (row + col sums of B)
    dist_centrality = b.sum(axis=1) + b.sum(axis=0)
    
    # Surrogate cost: we want to assign high-flow facilities to low-distance locations
    # Create a cost matrix where cost[i, j] = -flow_centrality[i] * dist_centrality[j]
    # (negative because we want to minimize, and high flow * low dist = good)
    surrogate = -np.outer(flow_centrality, dist_centrality)
    
    row_ind, col_ind = linear_sum_assignment(surrogate)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _sa_polish(a, b, p, cost, time_limit, start_time):
    """
    Simulated annealing polish with geometric cooling.
    """
    n = len(p)
    p_sa = p.copy()
    cost_sa = cost
    best_p = p_sa.copy()
    best_cost = cost_sa
    
    T = 50.0
    T_min = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit and T > T_min:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p_sa, i, j)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
            cost_sa += delta
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
        
        T *= cooling
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Step 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    # Step 2: Best-improvement 2-opt local search
    elapsed = time.time() - start_time
    ls_time = min(60.0, total_deadline - elapsed - 30)  # Leave 30s for SA
    if ls_time > 1.0:
        p_ls, cost_ls = _best_improvement_2opt(a, b, p.copy(), ls_time, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Step 3: SA polish
    elapsed = time.time() - start_time
    sa_time = total_deadline - elapsed - 2
    if sa_time > 1.0:
        p_sa, cost_sa = _sa_polish(a, b, best_p.copy(), best_cost, sa_time, start_time)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    if best_p is None:
        best_p = np.arange(n)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
