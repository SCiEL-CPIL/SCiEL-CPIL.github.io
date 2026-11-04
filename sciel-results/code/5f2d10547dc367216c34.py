"""
Hybrid Spectral QL + Vectorized SA + Tabu Search
Strategy:
1. Use Spectral Quadratic Linearization (QL) to generate a high-quality initial solution.
2. Employ a vectorized Simulated Annealing (SA) phase for broad exploration.
3. Introduce a Tabu Search (TS) phase with a dynamic tabu list to escape local optima more effectively than pure LS.
4. Use a final high-precision polish pass with a tighter tolerance and more iterations.
5. Keep the multi-restart structure but allocate time more dynamically based on observed progress.
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
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _spectral_ql_init(a, b, num_components=15):
    """
    Quadratic Linearization via Spectral Decomposition.
    """
    n = len(a)
    
    # Symmetrize A for spectral decomposition stability
    A_sym = (a + a.T) / 2.0
    
    # Compute top eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    
    # Sort by magnitude of eigenvalues (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    best_p = None
    best_cost = np.inf
    
    # Try a few different numbers of components
    for k in range(1, min(num_components + 1, n)):
        V_k = eigvecs[:, :k]
        Lambda_k = eigvals[:k]
        
        # proj_b shape: (k, n)
        proj_b = V_k.T @ b  # (k, n)
        
        # C[i, j] = sum_{m=1}^k Lambda_k[m] * V_k[i, m] * proj_b[m, j]
        C = (V_k * Lambda_k) @ proj_b
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p, best_cost

def _sa_polish(a, b, p, time_limit):
    """Simulated annealing polish phase with vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Cooling schedule
    t0 = 5000.0
    t_final = 0.01
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        # Process a batch of swaps to amortize time check
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
                
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Acceptance
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def _tabu_search(a, b, p, time_limit, tabu_len=20):
    """
    Tabu Search with a dynamic tabu list.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Tabu list: store (i, j) pairs that are forbidden for the next 'tabu_len' iterations
    tabu_list = {}
    iteration = 0
    
    # Precompute all pairs for faster access
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_delta = np.inf
        best_i, best_j = -1, -1
        
        # Shuffle pairs to avoid bias
        np.random.shuffle(pairs)
        
        for i, j in pairs:
            # Check if this move is tabu
            if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                continue
            if (j, i) in tabu_list and tabu_list[(j, i)] > iteration:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Aspiration criterion: accept tabu move if it improves the best solution
            if delta < best_delta or (delta < best_cost - current_cost and delta < best_delta):
                best_delta = delta
                best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        # Apply the best move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu list
        tabu_list[(best_i, best_j)] = iteration + tabu_len
        tabu_list[(best_j, best_i)] = iteration + tabu_len
        
        # Clean up old tabu entries
        if iteration % tabu_len == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _local_search(a, b, p, time_limit):
    """First-improvement local search."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Shuffle indices to ensure we don't always start from (0,1)
        indices = np.random.permutation(n)
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                i = indices[i_idx]
                j = indices[j_idx]
                
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
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Spectral QL Initialization
    p_init, cost_init = _spectral_ql_init(a, b, num_components=15)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Multi-Restart SA + Tabu Search
    # We have ~110s left.
    # Let's do 4 restarts to allow more time per restart.
    num_restarts = 4
    time_per_restart = (deadline - (time.time() - start) - 5.0) / num_restarts
    time_per_restart = max(10.0, min(25.0, time_per_restart))
    
    for r in range(num_restarts):
        if time.time() - start > deadline - 10.0:
            break
            
        # Perturb current best
        p_current = best_p.copy()
        # Perform 10-20 random swaps
        num_swaps = 10 + np.random.randint(10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # SA Polish
        sa_time = time_per_restart * 0.5
        p_sa, cost_sa = _sa_polish(a, b, p_current, sa_time)
        
        # Tabu Search
        ts_time = time_per_restart * 0.3
        p_ts, cost_ts = _tabu_search(a, b, p_sa, ts_time, tabu_len=20)
        
        # LS Polish
        ls_time = time_per_restart * 0.2
        p_ls, cost_ls = _local_search(a, b, p_ts, ls_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # Phase 3: Final Deep Polish on the very best solution found
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5.0:
        p_final, cost_final = _local_search(a, b, best_p, time_remaining - 2.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
