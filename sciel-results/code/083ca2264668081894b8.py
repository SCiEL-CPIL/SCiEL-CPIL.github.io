"""
Simulated Annealing solver for QAPLIB wil100.
Uses vectorized delta computation for swaps with a multi-start and
annealing schedule that balances exploration and exploitation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search(a, b, p, time_limit, start_time, max_iter=None):
    """
    Perform steepest descent local search until no improving move is found
    or limits are reached.
    Returns the improved permutation and cost.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 100000
    
    improved = True
    while improved and max_iter > 0:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Check all pairs for the best improving move
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
            max_iter -= 1
    
    return p, current_cost

def _sa_with_tabu(a, b, p, current_cost, time_limit, start_time, tabu_len=25):
    """
    Simulated Annealing with tabu search to escape local minima.
    Returns the best permutation and cost found.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu list: store (i, j, expiry_iter) for recently made moves
    tabu_list = []
    tabu_set = set()
    iteration = 0
    
    T_start = max(1.0, current_cost * 0.01)
    T_end = 0.01
    total_duration = time_limit - (time.time() - start_time)
    if total_duration <= 0:
        return best_p, best_cost
    
    # Track stall counter
    stall_count = 0
    stall_threshold = 400
    
    while time.time() - start_time < time_limit - 3:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Check if move is tabu
        move_key = (min(i, j), max(i, j))
        is_tabu = move_key in tabu_set
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        remaining = time_limit - 3 - elapsed
        progress = 1.0 - (remaining / total_duration) if total_duration > 0 else 1.0
        T = T_start * (T_end / T_start) ** progress
        
        # Accept if improving, or if not tabu and SA accepts
        accept = False
        if delta < 0:
            accept = True
        elif not is_tabu and T > 0.001:
            if np.random.rand() < np.exp(-delta / T):
                accept = True
        # Aspiration: accept tabu move if it improves best
        elif is_tabu and new_cost < best_cost:
            accept = True
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            
            # Add to tabu list
            expiry = iteration + tabu_len
            tabu_list.append((move_key, expiry))
            tabu_set.add(move_key)
            
            # Remove expired tabu entries
            while tabu_list and tabu_list[0][1] <= iteration:
                old_key, _ = tabu_list.pop(0)
                tabu_set.discard(old_key)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # If stalled for too long, clear tabu to allow exploration
            if stall_count > stall_threshold:
                tabu_list.clear()
                tabu_set.clear()
                stall_count = 0
        
        iteration += 1
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 140.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search(a, b, p, time_limit=total_deadline - 30, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        # Try a few random starts
        num_random_starts = min(4, int(remaining_time // 15))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 30:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search(a, b, p, time_limit=total_deadline - 30, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA + tabu on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 5 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        # Use SA with tabu for the final polish
        best_p, best_cost = _sa_with_tabu(a, b, p, current_cost, total_deadline, start_time, tabu_len=25)
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
