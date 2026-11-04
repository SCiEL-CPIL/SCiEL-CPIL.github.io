"""
Combined Tabu Search + SA Polish solver for QAPLIB wil100.
Strategy:
1. Strong initial solution via greedy rank-matching.
2. Vectorized Tabu Search with O(n^2) swap evaluation per iteration.
3. Simulated Annealing polish for fine-tuning.
4. Diversification restarts if time permits.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

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

def _tabu_search(a, b, p_init, time_limit, tabu_tenure=10, seed=None):
    """
    Perform Tabu Search starting from p_init.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    tabu_list = {} # position -> iteration when it became tabu
    current_iter = 0
    
    while time.time() - start_time < time_limit:
        best_move_i = -1
        best_move_j = -1
        best_move_cost = np.inf
        
        # Evaluate all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                pi = p[i]
                pj = p[j]
                
                # Check tabu status
                is_tabu = False
                if i in tabu_list and current_iter - tabu_list[i] < tabu_tenure:
                    is_tabu = True
                if j in tabu_list and current_iter - tabu_list[j] < tabu_tenure:
                    is_tabu = True
                
                # Calculate delta using efficient O(n) computation
                delta = _swap_delta(a, b, p, i, j)
                new_cost = current_cost + delta
                
                # Aspiration criterion: if new_cost < best_cost, it's allowed even if tabu
                if is_tabu and new_cost >= best_cost:
                    continue
                
                if new_cost < best_move_cost:
                    best_move_cost = new_cost
                    best_move_i = i
                    best_move_j = j
        
        # If no valid move found, break
        if best_move_i == -1:
            break
        
        # Perform the move
        p[best_move_i], p[best_move_j] = p[best_move_j], p[best_move_i]
        current_cost = best_move_cost
        
        # Update tabu list
        tabu_list[best_move_i] = current_iter
        tabu_list[best_move_j] = current_iter
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        current_iter += 1
        
        # Clean up old tabu entries
        if current_iter > 100:
            keys_to_remove = [k for k, v in tabu_list.items() if current_iter - v >= tabu_tenure]
            for k in keys_to_remove:
                del tabu_list[k]
        
        if time.time() - start_time > time_limit:
            break
            
    return best_p, best_cost

def _sa_polish(a, b, p, time_limit, seed=None):
    """Simulated annealing polish phase with vectorized delta."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
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
        
        # Process a batch of swaps
        for _ in range(30):
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
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    # Deadline: 150s total. Leave 10s for overhead/safety.
    total_deadline = 140.0
    start_time = time.time()
    
    # 1. Initial Solution
    p_init = _greedy_initial(a, b)
    best_p = p_init
    best_cost = _qap_cost(a, b, p_init)
    
    # 2. Main Tabu Search - allocate 60% of time
    time_for_tabu = total_deadline * 0.6
    p_tabu, cost_tabu = _tabu_search(a, b, p_init, time_for_tabu, tabu_tenure=10)
    
    if cost_tabu < best_cost:
        best_p = p_tabu
        best_cost = cost_tabu
    
    # 3. SA Polish - allocate 25% of time
    time_remaining = total_deadline - (time.time() - start_time)
    time_for_sa = min(total_deadline * 0.25, time_remaining * 0.7)
    
    if time_for_sa > 2:
        p_sa, cost_sa = _sa_polish(a, b, best_p, time_for_sa)
        
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa
    
    # 4. Diversification / Restart - use remaining time
    time_remaining = total_deadline - (time.time() - start_time)
    
    if time_remaining > 5:
        # Try 2 restarts from perturbed best solution
        num_restarts = 2
        time_per_restart = (time_remaining - 2) / num_restarts
        
        for r in range(num_restarts):
            if time.time() - start_time > total_deadline - 1:
                break
                
            p_restart = best_p.copy()
            # Perform random swaps to diversify
            num_swaps = 5 + r * 3
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
                
            # Run tabu search from this perturbed state
            p_rs, cost_rs = _tabu_search(a, b, p_restart, time_per_restart, tabu_tenure=10)
            
            if cost_rs < best_cost:
                best_p = p_rs
                best_cost = cost_rs
                
                # Quick SA polish if time permits
                time_left = total_deadline - (time.time() - start_time)
                if time_left > 3:
                    p_sa2, cost_sa2 = _sa_polish(a, b, best_p, min(3.0, time_left - 1))
                    if cost_sa2 < best_cost:
                        best_p = p_sa2
                        best_cost = cost_sa2
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
