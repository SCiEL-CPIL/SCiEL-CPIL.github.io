"""Search-based solver for QAPLIB wil100: Hungarian Init + Vectorized SA with adaptive restarts."""
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
    
    # Diagonal terms (usually 0 for QAP but handle generally)
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    
    # Compute over all k, then subtract k=i and k=j contributions
    
    # Term 1 over all k:
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Subtract k=j: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2 over all k:
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Subtract k=j: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Uses row-sum centrality to match high-flow facilities with central locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Normalize row sums
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums
        
    # Surrogate cost: minimize -a_row_sums[i] * b_row_sums[j]
    # i.e., maximize a_row_sums[i] * b_row_sums[j]
    C = -np.outer(ars_norm, brs_norm)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _sa_polish(a, b, p, time_limit, t0=30000.0, t_final=0.05):
    """
    Simulated annealing polish phase with vectorized delta and adaptive restarts.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Adaptive restart mechanism
    no_improve_count = 0
    max_no_improve = 50000
    restart_count = 0
    
    # Batch processing for efficiency
    batch_size = 50
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        for _ in range(batch_size):
            if time.time() - start_time > time_limit:
                break
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Acceptance criterion
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Adaptive restart: if stuck for too long, reset to best solution
        if no_improve_count > max_no_improve:
            p = best_p.copy()
            current_cost = best_cost
            no_improve_count = 0
            restart_count += 1
            # Increase temperature slightly on restart to escape local minima
            t0 = min(t0 * 1.5, 100000.0)
    
    return best_p, best_cost

def _local_search(a, b, p, time_limit):
    """First-improvement local search for fast convergence."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
    
    return best_p, best_cost

def _perturb(p, k):
    """Perturb permutation by k random swaps."""
    n = len(p)
    p = p.copy()
    for _ in range(k):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    deadline = 110.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    # Strategy 1: Hungarian initial solution
    p_hung = _hungarian_initial(a, b)
    cost_hung = _qap_cost(a, b, p_hung)
    
    best_p = p_hung.copy()
    best_cost = cost_hung
    
    # Strategy 2: Local search from Hungarian solution
    ls_time = min(5.0, (deadline - (time.time() - start)) * 0.2)
    if ls_time > 1:
        p_ls, cost_ls = _local_search(a, b, p_hung.copy(), ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Strategy 3: ILSA - Iterated Local Search with SA polish and adaptive restarts
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() - start > deadline - 10:
            break
        
        # Choose starting point
        if restart == 0:
            p_curr = best_p.copy()
        else:
            # Perturb the best solution with increasing perturbation strength
            k = 2 + restart % 8
            p_curr = _perturb(best_p, k)
        
        # Local search
        time_remaining = deadline - (time.time() - start) - 5
        if time_remaining < 2:
            break
        
        # Allocate time between LS and SA
        ls_time = min(2.0, time_remaining * 0.3)
        sa_time = min(3.0, time_remaining * 0.5)
        
        p_ls, cost_ls = _local_search(a, b, p_curr.copy(), ls_time)
        
        # SA polish with adaptive restarts
        p_sa, cost_sa = _sa_polish(a, b, p_ls.copy(), sa_time)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    # Final long SA run from best solution with higher temperature
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 15:
        p_final, cost_final = _sa_polish(a, b, best_p.copy(), time_remaining - 5, t0=50000.0)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
