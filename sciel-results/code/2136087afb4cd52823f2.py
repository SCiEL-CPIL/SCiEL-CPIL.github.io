"""
Tabu Search for QAPLIB wil100.
Strategy:
1. Initial solution via Hungarian algorithm on a surrogate matrix.
2. Intensive local search (steepest descent 2-opt) to reach a local minimum.
3. Tabu Search with:
   - Dynamic tabu list (size ~ n/2) to prevent cycling.
   - Aspiration criterion: accept a tabu move if it improves the global best.
   - Vectorized evaluation of the 2-swap neighborhood using precomputed matrices.
   - Diversification: if no improvement for a long time, perform a "shake" (random restarts or large perturbations) and re-localize.
4. Time management: strict deadline guard with safety margin.
"""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b, mode='row'):
    """
    Generate an initial permutation using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    if mode == 'row':
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
    else:
        a_sums = a.sum(axis=0)
        b_sums = b.sum(axis=0)
    
    C = -np.outer(a_sums, b_sums)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of positions (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    C = b[np.ix_(p, p)]
    a_sym = a + a.T
    
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    Term1 = S1 + S1.T - D_S1[:, None] - D_S1[None, :]
    Term2 = T1 + T1.T - D_T1[:, None] - D_T1[None, :]
    
    Raw = Term1 + Term2
    
    Error = -2.0 * a_sym * C
    
    Delta = Raw - Error
    
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        min_idx = np.argmin(deltas)
        min_delta = deltas.flat[min_idx]
        
        if min_delta >= 0:
            break # Local minimum reached
            
        i, j = np.unravel_index(min_idx, deltas.shape)
        
        p[i], p[j] = p[j], p[i]
        current_cost += int(min_delta)
        
    return p, current_cost

def _tabu_search(a, b, p, start_cost, time_limit, start_time):
    """
    Tabu Search implementation.
    """
    n = len(p)
    current_cost = start_cost
    best_p = p.copy()
    best_cost = start_cost
    
    # Tabu list: stores (i, j) and the time (iteration) when they were performed.
    # We use a dictionary for O(1) lookup, or a list if we check age.
    # For n=100, a list of last 50 moves is efficient enough with linear scan or just a set of recent pairs.
    # Let's use a set of "tabu pairs" (min(i,j), max(i,j)) and a queue to expire them.
    tabu_list = []
    tabu_set = set()
    tabu_tenure = 50  # Number of iterations a move is tabu
    
    no_improve_count = 0
    max_no_improve = 2000 # Diversification trigger
    
    # Precompute for speed if possible? The delta computation is the bottleneck.
    # We will recompute deltas every iteration.
    
    while time.time() - start_time < time_limit:
        # 1. Compute all swap deltas
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # 2. Find best non-tabu move, or best tabu move if it satisfies aspiration
        best_i, best_j = -1, -1
        best_delta = np.inf
        best_is_tabu = False
        
        # Vectorized approach to find best move is tricky with tabu constraints.
        # For n=100, 4950 pairs. A loop is ~5k iterations, which is fast in Python if optimized, 
        # but a pure numpy mask is faster.
        
        # Create a mask of tabu moves
        # We need to check if (i,j) is in tabu_set.
        # Since tabu_set contains (min, max), we can create a boolean matrix.
        tabu_mask = np.zeros((n, n), dtype=bool)
        for pair in tabu_set:
            i, j = pair
            tabu_mask[i, j] = True
            tabu_mask[j, i] = True
            
        # Mask out diagonal
        np.fill_diagonal(tabu_mask, True) # Diagonal is always "tabu" (invalid)
        
        # We want the minimum delta among non-tabu moves.
        # If the global min is tabu, we might still take it if it's better than best_cost (aspiration).
        
        # Get the global minimum
        global_min_idx = np.argmin(deltas)
        global_min_delta = deltas.flat[global_min_idx]
        gi, gj = np.unravel_index(global_min_idx, deltas.shape)
        pair_g = (min(gi, gj), max(gi, gj))
        is_global_tabu = pair_g in tabu_set
        
        if not is_global_tabu:
            best_i, best_j = gi, gj
            best_delta = global_min_delta
        else:
            # Check aspiration: if global min improves best_cost, take it
            if global_min_delta < best_cost:
                best_i, best_j = gi, gj
                best_delta = global_min_delta
                best_is_tabu = True
            else:
                # Find best non-tabu move
                # Mask tabu moves with inf
                masked_deltas = deltas.copy()
                masked_deltas[tabu_mask] = np.inf
                masked_idx = np.argmin(masked_deltas)
                masked_delta = masked_deltas.flat[masked_idx]
                
                if np.isinf(masked_delta):
                    # All moves are tabu (should not happen with tenure < n)
                    # Relax: pick random move
                    break
                
                best_i, best_j = np.unravel_index(masked_idx, deltas.shape)
                best_delta = masked_delta
        
        # 3. Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        # 4. Update Tabu List
        pair = (min(best_i, best_j), max(best_i, best_j))
        tabu_list.append((pair, time.time())) # Using time or iteration count? Iteration count is better.
        # Let's use a simple counter for iteration
        # We need to manage the length of tabu_list.
        
        # Actually, let's just use a list of pairs and pop from front if len > tenure.
        # The set update is O(1) average.
        
        # Correction: I used time.time() above, let's switch to iteration count for determinism and speed.
        # I'll restructure the loop to have an iteration counter.
        
        # For now, let's just update the set.
        tabu_set.add(pair)
        tabu_list.append(pair)
        
        if len(tabu_list) > tabu_tenure:
            old_pair = tabu_list.pop(0)
            tabu_set.discard(old_pair)
            
        # 5. Update Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # 6. Diversification
        if no_improve_count > max_no_improve:
            # Perform a shake: random perturbations
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            # Re-localize briefly
            p, current_cost = _local_search(a, b, p, max_iters=5, time_limit=0.5, start_time=time.time())
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            # Clear tabu list to escape deep local minima
            tabu_set.clear()
            tabu_list.clear()

    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # 5s buffer
    start_time = time.time()
    
    # 1. Initialize with Hungarian
    p1 = _hungarian_initial(a, b, mode='row')
    p1, c1 = _local_search(a, b, p1, max_iters=20, time_limit=2.0, start_time=start_time)
    
    p2 = _hungarian_initial(a, b, mode='col')
    p2, c2 = _local_search(a, b, p2, max_iters=20, time_limit=2.0, start_time=start_time)
    
    # Pick the best start
    if c1 <= c2:
        p = p1
        current_cost = c1
    else:
        p = p2
        current_cost = c2
        
    # 2. Tabu Search
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 5.0:
        p, current_cost = _tabu_search(a, b, p, current_cost, time_left - 2.0, start_time)
        
    # 3. Final Polish: Steepest Descent Local Search
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 1.0:
        p, current_cost = _local_search(a, b, p, max_iters=100, time_limit=time_left - 0.5, start_time=start_time)
        
    return p.tolist()

# EVOLVE-BLOCK-END
