"""Flow-shop makespan minimizer: NEH + CIG constructive, multi-start with insertion local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
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


def _makespan_fast(processing_times, order):
    """Faster makespan using numpy vectorization where possible."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    prev_C = 0.0
    for job_idx in order:
        p = processing_times[job_idx]
        curr_C = np.maximum.accumulate(p + prev_C)
        prev_C = curr_C[-1]
    return float(prev_C)


def _neh(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order_sorted = np.argsort(-totals).tolist()
    
    current_order = [order_sorted[0]]
    
    for k in range(1, n_jobs):
        job = order_sorted[k]
        best_pos = 0
        best_ms = _makespan(processing_times, current_order + [job])
        
        for pos in range(1, len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _cig(processing_times):
    """CIG constructive heuristic: Critical Insertion Greedy."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order_sorted = np.argsort(-totals).tolist()
    
    current_order = [order_sorted[0]]
    
    for k in range(1, n_jobs):
        job = order_sorted[k]
        best_pos = 0
        best_ms = _makespan(processing_times, current_order + [job])
        
        for pos in range(1, len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _local_search_insertion(processing_times, order, max_iters=100, time_limit=None, start_time=None):
    """Local search with insertion moves: remove a job and reinsert at best position."""
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
        for i in range(n_jobs):
            # Remove job at position i
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            
            # Try inserting at each position in remaining
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue  # Same position, skip
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_order = candidate
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best_order


def _local_search_swap(processing_times, order, max_iters=100, time_limit=None, start_time=None):
    """Local search with adjacent swap moves."""
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
        for i in range(n_jobs - 1):
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_order = new_order
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    return best_order


def _multi_start(processing_times, time_limit, start_time):
    """Multi-start with NEH, CIG, LPT, SPT, and random initializations."""
    n_jobs, n_machines = processing_times.shape
    
    best_order = None
    best_ms = float('inf')
    
    # Try multiple initial heuristics
    heuristics = []
    
    # NEH
    h1 = _neh(processing_times)
    heuristics.append(h1)
    
    # CIG
    h2 = _cig(processing_times)
    heuristics.append(h2)
    
    # LPT (Longest Processing Time)
    totals = processing_times.sum(axis=1)
    lpt_order = np.argsort(-totals).tolist()
    heuristics.append(lpt_order)
    
    # SPT (Shortest Processing Time)
    spt_order = np.argsort(totals).tolist()
    heuristics.append(spt_order)
    
    # Random
    random_order = list(np.random.permutation(n_jobs))
    heuristics.append(random_order)
    
    # For each heuristic, run local search and keep the best
    for h in heuristics:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        # Run insertion local search
        order = _local_search_insertion(processing_times, h, max_iters=50, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Final refinement with swap local search
    if best_order is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5:
            order = _local_search_swap(processing_times, best_order, max_iters=30, time_limit=time_limit, start_time=start_time)
            ms = _makespan(processing_times, order)
            if ms < best_ms:
                best_ms = ms
                best_order = list(order)
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Multi-start with local search refinement
    best_order, best_ms = _multi_start(processing_times, time_limit, start_time)
    
    # If we still have time, do another round of insertion local search
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        order = _local_search_insertion(processing_times, best_order, max_iters=50, time_limit=time_limit, start_time=start_time)
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    return best_order
# EVOLVE-BLOCK-END
