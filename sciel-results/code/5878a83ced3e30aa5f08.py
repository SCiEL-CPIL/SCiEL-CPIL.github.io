"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    c_prev = [0.0] * n_machines
    c_curr = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[-1]

def _compute_C(processing_times, order):
    """Compute the full completion time matrix C for the given order."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    C = [[0.0] * n_machines for _ in range(n_jobs)]
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                C[i][j] = p
            elif i == 0:
                C[i][j] = C[i][j-1] + p
            elif j == 0:
                C[i][j] = C[i-1][j] + p
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + p
    
    return C

def _delta_swap(processing_times, C, order, i, j):
    """
    Compute the change in makespan resulting from swapping jobs at positions i and j.
    Only recomputes the rows between min(i,j) and max(i,j).
    Returns the new makespan.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return C[n_jobs-1][n_machines-1]
    
    lo, hi = min(i, j), max(i, j)
    
    # We need to reconstruct the C values for rows lo to hi.
    # The values for rows 0 to lo-1 remain unchanged.
    # The values for rows hi+1 to n_jobs-1 depend on C[hi], so they would change, 
    # but for the purpose of evaluating the move's immediate impact on the final makespan,
    # we must propagate the changes all the way to the end.
    # Wait, the delta evaluation for flow shop is tricky because a change at row k 
    # affects all subsequent rows. 
    # However, a common optimization is that if we only care about the *last* element, 
    # we can just simulate the rows from lo to n_jobs-1.
    # Actually, recomputing from lo to n_jobs-1 is O((N-lo)M).
    # If we recompute only lo to hi, we don't know the final makespan unless we propagate.
    # So, we recompute rows from lo to n_jobs-1.
    
    # Copy the C matrix up to lo-1
    # We don't need to copy the whole C, just keep the state.
    # Let's create a temporary list of rows from lo to n_jobs-1.
    
    # The "previous row" for row lo is C[lo-1] (or zeros if lo==0)
    prev_row = C[lo-1] if lo > 0 else [0.0] * n_machines
    
    # We will compute the new values for rows lo to n_jobs-1
    # We need the order of jobs in this segment.
    # The jobs at lo and hi are swapped.
    
    # Let's build the sequence of job indices for rows lo to n_jobs-1
    # Original: order[lo], ..., order[hi], ...
    # New: order[hi] at lo, ..., order[lo] at hi, ...
    
    new_ms = 0.0
    curr_row = [0.0] * n_machines
    
    for row in range(lo, n_jobs):
        if row == lo:
            job_idx = order[hi]
        elif row == hi:
            job_idx = order[lo]
        else:
            job_idx = order[row]
            
        p_row = processing_times[job_idx]
        
        for col in range(n_machines):
            p = p_row[col]
            if row == 0 and col == 0:
                curr_row[col] = p
            elif row == 0:
                curr_row[col] = curr_row[col-1] + p
            elif col == 0:
                curr_row[col] = prev_row[col] + p
            else:
                curr_row[col] = max(prev_row[col], curr_row[col-1]) + p
        
        if row == n_jobs - 1:
            new_ms = curr_row[n_machines - 1]
            
        prev_row = curr_row
        curr_row = [0.0] * n_machines # Reset for next row? No, we need to be careful with references.
        # Actually, we can just use two lists and swap them.
        # But since we are inside a loop that depends on prev_row, let's just use a new list or swap.
        # Let's optimize this inner loop.
        
    return new_ms

def _tabu_search(processing_times, initial_order, time_limit=100.0):
    """
    Tabu search for flow shop scheduling.
    """
    start_time = time.time()
    n_jobs = len(initial_order)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return initial_order, _makespan(processing_times, initial_order)
    
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    current_C = _compute_C(processing_times, current_order)
    
    best_order = list(current_order)
    best_ms = current_ms
    best_C = [row[:] for row in current_C]
    
    # Tabu list: stores (i, j) pairs. 
    # We store the move to prevent immediate reversal.
    # A simple list of (i, j) with expiration.
    tabu_list = []
    tabu_tenure = max(10, n_jobs // 2) # Dynamic tenure
    
    # Aspiration: if a tabu move improves best_ms, it is allowed.
    
    no_improve_count = 0
    max_no_improve = n_jobs * 5 # Restart if stuck
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        best_move = None
        best_new_ms = current_ms
        best_i, best_j = -1, -1
        best_is_tabu = False
        
        # Evaluate all possible swaps
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Check if move is tabu
                is_tabu = False
                # Check if (i,j) or (j,i) is in tabu list
                # Since we store (min, max), just check (min(i,j), max(i,j))
                move_key = (min(i, j), max(i, j))
                for t_move in tabu_list:
                    if t_move[0] == move_key:
                        is_tabu = True
                        break
                
                # Calculate new makespan
                # To save time, we can do a quick check or just compute.
                # Computing delta is O(NM) in worst case (if i=0, j=N-1).
                # But average is better.
                
                new_ms = _delta_swap(processing_times, current_C, current_order, i, j)
                
                # Aspiration criterion
                if is_tabu and new_ms >= best_ms:
                    continue
                
                if new_ms < best_new_ms:
                    best_new_ms = new_ms
                    best_move = (i, j)
                    best_i, best_j = i, j
                    best_is_tabu = is_tabu
        
        if best_move is None:
            # No improving move found (or all were tabu and not aspirating)
            # Try a random move or restart
            if no_improve_count > max_no_improve:
                # Perturb: random swap
                i, j = random.sample(range(n_jobs), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
                current_ms = _makespan(processing_times, current_order)
                current_C = _compute_C(processing_times, current_order)
                tabu_list.clear()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, try a random non-tabu move even if it worsens? 
                # Or just break and rely on the final best.
                # For simplicity and speed, let's just break if we can't improve.
                # But tabu search usually accepts worsening moves to escape.
                # Let's accept the best move even if it's worse than current, 
                # unless it's strictly worse than best and tabu.
                
                # Re-evaluate: find the best move regardless of improvement, 
                # but respecting tabu (unless aspiration).
                best_move = None
                best_new_ms = float('inf')
                best_i, best_j = -1, -1
                best_is_tabu = False
                
                for i in range(n_jobs):
                    if time.time() - start_time > time_limit:
                        break
                    for j in range(i + 1, n_jobs):
                        if time.time() - start_time > time_limit:
                            break
                        move_key = (min(i, j), max(i, j))
                        is_tabu = False
                        for t_move in tabu_list:
                            if t_move[0] == move_key:
                                is_tabu = True
                                break
                        
                        if is_tabu and new_ms >= best_ms: # Note: new_ms is from previous loop, need to recalc
                            continue
                        
                        # Recalculate for this specific move
                        new_ms = _delta_swap(processing_times, current_C, current_order, i, j)
                        
                        if is_tabu and new_ms >= best_ms:
                            continue
                            
                        if new_ms < best_new_ms:
                            best_new_ms = new_ms
                            best_move = (i, j)
                            best_i, best_j = i, j
                            best_is_tabu = is_tabu
                
                if best_move is None:
                    # All moves are tabu and not aspirating
                    break
                    
                # Accept the move
                current_order[best_i], current_order[best_j] = current_order[best_j], current_order[best_i]
                current_ms = best_new_ms
                current_C = _compute_C(processing_times, current_order)
                
                # Update tabu list
                tabu_list.append(((min(best_i, best_j), max(best_i, best_j)), time.time() + tabu_tenure * 0.001))
                # Clean old tabu entries
                current_t = time.time()
                tabu_list = [t for t in tabu_list if t[1] > current_t]
                
                no_improve_count = 0
                
        else:
            # Apply the best improving move
            current_order[best_i], current_order[best_j] = current_order[best_j], current_order[best_i]
            current_ms = best_new_ms
            current_C = _compute_C(processing_times, current_order)
            
            # Update tabu list
            tabu_list.append(((min(best_i, best_j), max(best_i, best_j)), time.time() + tabu_tenure * 0.001))
            current_t = time.time()
            tabu_list = [t for t in tabu_list if t[1] > current_t]
            
            # Update best
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
                best_C = [row[:] for row in current_C]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    return best_order, best_ms

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    
    # Run Tabu Search
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Allocate time to tabu search
    time_limit = remaining_time - 5.0
    
    best_order, best_ms = _tabu_search(
        processing_times, initial_order, 
        time_limit=time_limit
    )
    
    return best_order
# EVOLVE-BLOCK-END
