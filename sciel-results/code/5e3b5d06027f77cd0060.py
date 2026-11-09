"""Combined program: Fast NEH variants with Tabu Search and efficient local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed using list-based row updates.
    """
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # Use lists for the state to minimize allocation overhead
    # prev_row[m] is completion time of previous job on machine m
    # curr_row[m] is completion time of current job on machine m
    prev_row = [0.0] * n_machines
    curr_row = [0.0] * n_machines
    
    for idx in range(n_jobs):
        job = order[idx]
        # Accessing numpy array row
        row = P[job]
        
        if idx == 0:
            s = 0.0
            for m in range(n_machines):
                s += row[m]
                curr_row[m] = s
        else:
            # Machine 0: only constrained by previous job on machine 0
            curr_row[0] = prev_row[0] + row[0]
            for m in range(1, n_machines):
                # Constrained by previous job on same machine (prev_row[m]) 
                # and current job on previous machine (curr_row[m-1])
                left = curr_row[m - 1]
                up = prev_row[m]
                mx = left if left > up else up
                curr_row[m] = mx + row[m]
        
        # Swap references to avoid copying
        prev_row, curr_row = curr_row, prev_row
        
    return prev_row[-1]

def _neh_heuristic(P, sort_key='total', seed=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1]), 'last' (P[:,-1])
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
    if sort_key == 'total':
        totals = P.sum(axis=1)
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'last':
        key_vals = P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan_fast(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # Insert at best position
        current_seq.insert(best_pos, job)
        
    return current_seq

def _local_search_insertion(order, P, deadline):
    """
    Local search using insertion moves (remove job, re-insert).
    Uses First Improvement strategy for speed.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan_fast(current_order, P)

    # We will loop until no improvement or deadline
    while True:
        if time.time() > deadline:
            break

        found_improvement = False
        # Iterate through jobs to remove
        for i in range(n_jobs):
            if time.time() > deadline:
                break

            job = current_order[i]
            # Remove job i
            remaining = current_order[:i] + current_order[i+1:]
            
            # Try all insertion positions in remaining
            # len(remaining) is n_jobs - 1
            # Positions 0 to n_jobs - 1
            for j in range(len(remaining) + 1):
                if time.time() > deadline:
                    break
                
                # Skip if inserting back at same position (j == i)
                if j == i:
                    continue

                new_order = remaining[:j] + [job] + remaining[j:]
                new_ms = _makespan_fast(new_order, P)
                
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    found_improvement = True
                    break # Restart the outer loop from i=0 (or just break to top of while)
            
            if found_improvement:
                break
        
        if not found_improvement:
            break

    return current_order

def _tabu_search(order, P, deadline):
    """
    Tabu Search with insertion moves.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan_fast(current_order, P)
    
    best_order = list(current_order)
    best_ms = current_ms

    # Tabu parameters
    # Tenure: how many iterations a move stays tabu
    tenure = max(5, n_jobs // 2)
    tabu_list = {}  # (job, pos) -> iteration when it becomes non-tabu
    iteration = 0
    no_improve_count = 0
    max_no_improve = 10 * n_jobs # Stop if stuck for a while
    
    while time.time() < deadline:
        iteration += 1
        if no_improve_count > max_no_improve:
            break
            
        best_move_ms = float('inf')
        best_move_i = -1
        best_move_j = -1
        best_move_is_tabu = False
        
        # Generate all insertion moves
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                
                # Check tabu status
                move_key = (job, j)
                # A move is tabu if the last time we made this specific (job, pos) move 
                # was within the last 'tenure' iterations.
                # We store the iteration number when the move was made.
                if move_key in tabu_list:
                    if iteration - tabu_list[move_key] < tenure:
                        is_tabu = True
                    else:
                        is_tabu = False
                else:
                    is_tabu = False
                
                new_order = remaining[:j] + [job] + remaining[j:]
                ms = _makespan_fast(new_order, P)
                
                # Aspiration criterion: accept tabu move if it improves global best
                if is_tabu and ms >= best_ms:
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move_is_tabu = is_tabu
        
        if best_move_i == -1:
            break
            
        # Apply the best move
        job = current_order[best_move_i]
        remaining = current_order[:best_move_i] + current_order[best_move_i+1:]
        current_order = remaining[:best_move_j] + [job] + remaining[best_move_j:]
        current_ms = best_move_ms
        
        # Update tabu list
        # The move we just made is (job, best_move_j). 
        # Note: The "reverse" move would be different, but typically we tabu the move just made 
        # to prevent oscillation.
        tabu_list[(job, best_move_j)] = iteration
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_order = list(current_order)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return best_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses multiple NEH variants, local search, and tabu search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    # Keep 15s safety margin before the 150s kill
    deadline = start_time + 135.0

    best_perm = None
    best_ms = float('inf')

    # 1. Generate diverse initial solutions using NEH variants
    # Variant 1: sort by total processing time
    perm1 = _neh_heuristic(P, sort_key='total')
    ms1 = _makespan_fast(perm1, P)
    if ms1 < best_ms:
        best_ms = ms1
        best_perm = perm1

    # Variant 2: sort by first + last machine processing time
    perm2 = _neh_heuristic(P, sort_key='first_last')
    ms2 = _makespan_fast(perm2, P)
    if ms2 < best_ms:
        best_ms = ms2
        best_perm = perm2
        
    # Variant 3: sort by last machine processing time
    perm3 = _neh_heuristic(P, sort_key='last')
    ms3 = _makespan_fast(perm3, P)
    if ms3 < best_ms:
        best_ms = ms3
        best_perm = perm3

    # Add a few randomized NEH restarts if time permits
    rng = random.Random(12345)
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        seed = rng.randint(0, 10**9)
        perm_r = _neh_heuristic(P, sort_key='total', seed=seed)
        ms_r = _makespan_fast(perm_r, P)
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r

    # 2. Apply Local Search (Fast, First Improvement)
    if time.time() < deadline - 10.0 and len(best_perm) > 1:
        improved_perm = _local_search_insertion(best_perm, P, deadline=deadline - 5.0)
        improved_ms = _makespan_fast(improved_perm, P)

        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms

    # 3. Apply Tabu Search to escape local optima
    if time.time() < deadline - 2.0 and len(best_perm) > 1:
        tabu_perm = _tabu_search(best_perm, P, deadline=deadline - 1.0)
        tabu_ms = _makespan_fast(tabu_perm, P)

        if tabu_ms < best_ms:
            best_perm = tabu_perm
            best_ms = tabu_ms

    return list(best_perm)

# EVOLVE-BLOCK-END
