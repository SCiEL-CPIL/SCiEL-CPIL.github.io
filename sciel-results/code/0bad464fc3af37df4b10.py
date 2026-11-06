"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _v_shape_initial(processing_times):
    """
    Generate an initial V-shaped permutation.
    
    Jobs are sorted by total processing time.
    The first half (most time-consuming) go to the front, 
    the second half (least time-consuming) go to the back in reverse order.
    This creates a V-shape with respect to job completion times.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs <= 2:
        return list(range(n_jobs))
    
    totals = processing_times.sum(axis=1)
    # Sort indices by total processing time descending
    sorted_indices = np.argsort(-totals)
    
    # Split into two halves
    half = n_jobs // 2
    front = sorted_indices[:half]
    back = sorted_indices[half:]
    
    # V-shape: front in order, back in reverse order
    # Actually, standard V-shape puts highest priority at start and end
    # Let's try: front jobs in sorted order, back jobs in sorted order but reversed placement?
    # Common heuristic: sort by total time. First k jobs go to start, last (n-k) go to end in reverse.
    # Let's do: front = first half of sorted, back = second half of sorted reversed
    # So the sequence is: [sorted[0], ..., sorted[half-1], sorted[n-1], ..., sorted[half]]
    
    perm = list(front) + list(back[::-1])
    return perm


def _makespan(processing_times, perm):
    """Calculate the makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    # C[j][0] = sum of processing times of jobs 0..j on machine 0
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    
    # Use a 1D array for current row and previous row
    prev_row = np.zeros(n_machines)
    curr_row = np.zeros(n_machines)
    
    makespan = 0.0
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                curr_row[0] = prev_row[0] + processing_times[j, 0]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[j, m]
        makespan = curr_row[-1]
        prev_row = curr_row.copy()
    
    return makespan


def _adjacent_swap_makespan_delta(processing_times, perm, i):
    """
    Calculate the new makespan after swapping adjacent jobs at positions i and i+1.
    Returns the new makespan.
    """
    # For efficiency, we could compute only the affected part, but for simplicity
    # and given time constraints, let's just recompute the makespan.
    # However, recomputing the full makespan is O(n*m) which might be slow for large n.
    # Let's try a more efficient approach if possible, but for now, correctness first.
    # Actually, let's just recompute. For n_jobs up to a few hundred and n_machines up to a few hundred,
    # this should be fine within 120s if we don't do too many iterations.
    
    new_perm = perm.copy()
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return _makespan(processing_times, new_perm)


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: Generate initial V-shaped solution
    current_perm = _v_shape_initial(processing_times)
    current_makespan = _makespan(processing_times, current_perm)
    
    # Step 2: Tabu Search
    # Dynamic tabu list: store (position, job) pairs or just positions?
    # Let's use a tabu list of (i, j) where we swapped i and j? 
    # Or simpler: tabu list of positions i where swapping i and i+1 is tabu.
    # Let's use a tabu list of integers representing the position i (swap i and i+1).
    
    import time
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer
    
    tabu_list = {}  # position -> iteration when it becomes non-tabu
    tabu_tenure = 10
    iteration = 0
    
    best_perm = current_perm.copy()
    best_makespan = current_makespan
    
    while time.time() - start_time < time_limit:
        iteration += 1
        improved = False
        
        # Find the best non-tabu move
        best_move_i = -1
        best_move_makespan = current_makespan
        best_move_perm = None
        
        # If no improvement found, allow tabu moves (aspiration criterion)
        best_tabu_move_i = -1
        best_tabu_move_makespan = current_makespan
        best_tabu_move_perm = None
        
        for i in range(n_jobs - 1):
            new_makespan = _adjacent_swap_makespan_delta(processing_times, current_perm, i)
            
            if new_makespan < best_move_makespan or (new_makespan == best_move_makespan and i < best_move_i):
                if i in tabu_list and tabu_list[i] > iteration:
                    # This move is tabu
                    if new_makespan < best_tabu_move_makespan:
                        best_tabu_move_makespan = new_makespan
                        best_tabu_move_i = i
                        best_tabu_move_perm = None  # Will be constructed later
                else:
                    best_move_makespan = new_makespan
                    best_move_i = i
                    best_move_perm = None  # Will be constructed later
            
            # Update best tabu move if it's better
            if i in tabu_list and tabu_list[i] > iteration:
                if new_makespan < best_tabu_move_makespan:
                    best_tabu_move_makespan = new_makespan
                    best_tabu_move_i = i
        
        # Select the move
        if best_move_i != -1:
            chosen_i = best_move_i
            chosen_makespan = best_move_makespan
        elif best_tabu_move_i != -1:
            # Aspiration: accept tabu move if it improves global best
            if best_tabu_move_makespan < best_makespan:
                chosen_i = best_tabu_move_i
                chosen_makespan = best_tabu_move_makespan
            else:
                # No good move, break or diversify
                # For now, break to avoid infinite loop
                break
        else:
            break
        
        # Apply the move
        new_perm = current_perm.copy()
        new_perm[chosen_i], new_perm[chosen_i + 1] = new_perm[chosen_i + 1], new_perm[chosen_i]
        
        current_perm = new_perm
        current_makespan = chosen_makespan
        
        # Update tabu list
        tabu_list[chosen_i] = iteration + tabu_tenure
        # Also tabu the reverse move? Swapping i and i+1 again would be at position i
        # So just tabu position i
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = current_perm.copy()
            # Diversify: increase tabu tenure?
            tabu_tenure = min(tabu_tenure + 1, 30)
        else:
            # No improvement, decrease tabu tenure to escape local optima
            tabu_tenure = max(tabu_tenure - 1, 5)
        
        # Clean up tabu list
        for key in list(tabu_list.keys()):
            if tabu_list[key] <= iteration:
                del tabu_list[key]
    
    return list(best_perm)
# EVOLVE-BLOCK-END
