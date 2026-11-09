"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # C[j] = completion time of current job on machine j
    # We process jobs in order, maintaining cumulative times per machine
    # C[j] after processing job i = max(C[j-1], prev_C[j]) + p[i][j]
    # where prev_C[j] is the completion time of the previous job on machine j
    
    prev_C = np.zeros(n_machines)
    for job_idx in order:
        curr_C = np.zeros(n_machines)
        for j in range(n_machines):
            if j == 0:
                curr_C[j] = prev_C[j] + processing_times[job_idx, j]
            else:
                curr_C[j] = max(curr_C[j-1], prev_C[j]) + processing_times[job_idx, j]
        prev_C = curr_C
    return prev_C[-1]


def _neh_rule(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order_sorted = np.argsort(-totals)
    
    # Start with first job
    current_order = [order_sorted[0]]
    
    for k in range(1, n_jobs):
        job = order_sorted[k]
        best_pos = 0
        best_ms = _makespan(processing_times, current_order + [job])
        
        # Try all possible insertion positions
        for pos in range(1, len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _local_search(processing_times, order, max_iters=200, time_limit=None, start_time=None):
    """2-opt local search: try swapping adjacent jobs, accept if improves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    no_improve_count = 0
    
    for _ in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit - 5:
                break
        
        improved = False
        # Try all adjacent swaps
        for i in range(n_jobs - 1):
            # Swap positions i and i+1
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_order = new_order
                improved = True
                no_improve_count = 0
                break  # Restart scan from beginning after improvement
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best_order


def _tabu_search(processing_times, order, max_iters=500, tabu_len=20, time_limit=None, start_time=None):
    """Tabu search with 2-opt moves (swap any two positions)."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    current_order = list(best_order)
    current_ms = best_ms
    
    # Tabu list: store (i, j) pairs that were recently swapped
    tabu_list = []
    tabu_dict = {}  # (i,j) -> iteration when it became tabu
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit - 5:
                break
        
        # Generate candidate moves
        best_move = None
        best_move_ms = current_ms
        best_move_order = None
        
        # Sample a subset of moves to keep it fast
        n_samples = min(50, n_jobs * (n_jobs - 1) // 2)
        if n_jobs * (n_jobs - 1) // 2 <= n_samples:
            pairs = [(i, j) for i in range(n_jobs) for j in range(i+1, n_jobs)]
        else:
            pairs = [(np.random.randint(n_jobs), np.random.randint(n_jobs)) for _ in range(n_samples)]
        
        for i, j in pairs:
            if i == j:
                continue
            # Check tabu
            key = (min(i,j), max(i,j))
            if key in tabu_dict and tabu_dict[key] > it - tabu_len:
                continue
            
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            ms = _makespan(processing_times, new_order)
            
            # Aspiration criterion: accept tabu move if it improves best
            if ms < best_move_ms or (key in tabu_dict and tabu_dict[key] <= it - tabu_len and ms < current_ms):
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (i, j)
                    best_move_order = new_order
        
        if best_move is None:
            # No non-tabu improving move; try any move
            i, j = np.random.randint(n_jobs), np.random.randint(n_jobs)
            if i == j:
                continue
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            ms = _makespan(processing_times, new_order)
            best_move = (i, j)
            best_move_ms = ms
            best_move_order = new_order
        
        # Apply best move
        current_order = best_move_order
        current_ms = best_move_ms
        i, j = best_move
        key = (min(i,j), max(i,j))
        tabu_dict[key] = it
        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)
        tabu_list.append(key)
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_order = list(current_order)
    
    return best_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial solution using NEH heuristic
    order = _neh_rule(processing_times)
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    
    # Local search refinement
    order = _local_search(processing_times, best_order, max_iters=100, time_limit=time_limit, start_time=start_time)
    ms = _makespan(processing_times, order)
    if ms < best_ms:
        best_ms = ms
        best_order = list(order)
    
    # Tabu search for further improvement
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        order = _tabu_search(processing_times, best_order, max_iters=300, tabu_len=15, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Final local search if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        order = _local_search(processing_times, best_order, max_iters=50, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    return best_order
# EVOLVE-BLOCK-END
