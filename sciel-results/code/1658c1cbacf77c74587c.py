"""
Hungarian Initialization + Vectorized Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Compute facility flow centrality (row sums of A) and location distance centrality (row sums of B).
2. Construct a linear surrogate cost matrix C[i, j] = -a_row_sums[i] * b_row_sums[j] 
   (minimizing C maximizes the product of centralities, aligning high-flow facilities 
   with high-distance locations in a globally optimal linear assignment).
3. Solve the assignment problem using the Hungarian algorithm (scipy.optimize.linear_sum_assignment) 
   to get a structurally coherent initial permutation p.
4. Perform Simulated Annealing with vectorized delta evaluation for 2-opt swaps.
   - Precompute row/col differences for fast delta calculation.
   - Use a cooling schedule that allows escape from local minima.
5. Return the best permutation found.
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
    
    # Cross terms with all other k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Surrogate cost: C[i, j] = -a_row_sums[i] * b_row_sums[j]
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    C = -np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _local_search(a, b, p, max_iters=1000):
    """
    First-improvement local search using 2-opt swaps.
    Returns improved permutation and cost.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
        if not improved:
            break
    return p, current_cost

def _perturb(a, b, p, k_swaps=5):
    """
    Apply k random swaps to perturb the permutation.
    """
    n = len(p)
    p_pert = p.copy()
    for _ in range(k_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
    return p_pert

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)  # Fixed seed for reproducibility within a run
    
    # Phase 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Iterated Local Search with SA acceptance
    T_start = 500.0
    T_end = 10.0
    
    p_ls = p.copy()
    cost_ls = current_cost
    
    while time.time() - start_time < total_deadline:
        # Local search to local optimum
        p_ls, cost_ls = _local_search(a, b, p_ls, max_iters=500)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Check time
        if time.time() - start_time > total_deadline - 10:
            break
        
        # Perturbation
        k_swaps = np.random.randint(3, 8)
        p_pert = _perturb(a, b, p_ls, k_swaps)
        cost_pert = _qap_cost(a, b, p_pert)
        
        # SA acceptance criterion
        frac = min(1.0, (time.time() - start_time) / total_deadline)
        T = T_start * (T_end / T_start) ** frac
        
        delta = cost_pert - cost_ls
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            p_ls = p_pert
            cost_ls = cost_pert
    
    # Final polish: SA from best solution
    p_sa = best_p.copy()
    cost_sa = best_cost
    T_start2 = 200.0
    T_end2 = 1.0
    
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed
    if remaining > 1:
        while time.time() - start_time < total_deadline:
            frac = min(1.0, (time.time() - start_time - (total_deadline - remaining)) / remaining)
            T = T_start2 * (T_end2 / T_start2) ** frac
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p_sa, i, j)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += int(delta)
                
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
