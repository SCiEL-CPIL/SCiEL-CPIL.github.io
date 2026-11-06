"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation using a fast list-based loop.
    This is typically faster than vectorized approaches for small M due to
    lower overhead and better cache locality for the small completion array.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        perm: list of job indices in order
    
    Returns:
        makespan (float)
    """
    if len(perm) == 0:
        return 0.0
        
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(np.sum(processing_times[perm]))
    
    # Local variables for speed
    P = processing_times
    completion = [0.0] * n_machines
    
    for job in perm:
        row = P[job]
        prev_c = 0.0
        for m in range(n_machines):
            c_curr = completion[m]
            if c_curr < prev_c:
                c_curr = prev_c
            c_curr += row[m]
            completion[m] = c_curr
            prev_c = c_curr
            
    return completion[-1]

def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence

def _v_shape_heuristic(processing_times):
    """
    V-shaped heuristic: 
    1. Sort jobs by total processing time descending.
    2. Split into two lists: first half (ascending) and second half (descending).
    3. Construct sequence by picking the best next job from the heads of these lists.
    This often yields good initial solutions for flow shop problems.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Split into two halves
    mid = n_jobs // 2
    list1 = list(order[:mid])
    list2 = list(order[mid:][::-1])
    
    sequence = []
    while list1 or list2:
        best_job = -1
        best_ms = float('inf')
        
        # Try head of list1
        if list1:
            job = list1[0]
            new_seq = sequence + [job]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_job = job
                best_source = 1
        
        # Try head of list2
        if list2:
            job = list2[0]
            new_seq = sequence + [job]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_job = job
                best_source = 2
        
        if best_job == -1:
            break
            
        if best_source == 1:
            list1.pop(0)
        else:
            list2.pop(0)
            
        sequence.append(best_job)
        
    return sequence

def _local_search(processing_times, initial_seq, deadline):
    """
    Greedy Local Search with time guards.
    Uses a windowed neighborhood to reduce complexity from O(N^2) to O(N*K).
    """
    n_jobs = len(initial_seq)
    if n_jobs < 2:
        return list(initial_seq), _calc_makespan(processing_times, initial_seq)
    
    current_seq = list(initial_seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    # Window size for moves: try moving job i to positions i-K to i+K
    # This reduces the number of evaluations significantly.
    window = min(n_jobs, 10)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                return best_seq, best_ms
            
            job = current_seq[i]
            # Determine range of positions to try
            start = max(0, i - window)
            end = min(n_jobs, i + window + 1)
            
            best_pos_in_window = i
            best_ms_in_window = current_ms
            
            for j in range(start, end):
                if j == i:
                    continue
                
                # Construct new sequence efficiently
                # Remove job at i
                new_seq = current_seq[:i] + current_seq[i+1:]
                # Insert at j (adjusted for removal)
                # If j > i, the index in the new list is j-1
                # If j <= i, the index is j
                insert_idx = j - 1 if j > i else j
                new_seq.insert(insert_idx, job)
                
                new_ms = _calc_makespan(processing_times, new_seq)
                
                if new_ms < best_ms_in_window:
                    best_ms_in_window = new_ms
                    best_pos_in_window = j
            
            # If we found a better position, apply the move
            if best_pos_in_window != i:
                # Apply the best move found in the window
                job = current_seq[i]
                new_seq = current_seq[:i] + current_seq[i+1:]
                insert_idx = best_pos_in_window - 1 if best_pos_in_window > i else best_pos_in_window
                new_seq.insert(insert_idx, job)
                
                current_seq = new_seq
                current_ms = best_ms_in_window
                improved = True
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seq = list(current_seq)
                    
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin to avoid timeout
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # NEH
    if time.time() < deadline - 5:
        candidates.append(_neh_heuristic(processing_times))
        
    # V-shaped
    if time.time() < deadline - 5:
        candidates.append(_v_shape_heuristic(processing_times))
        
    # Random
    if time.time() < deadline - 5:
        rand_perm = list(range(n_jobs))
        random.shuffle(rand_perm)
        candidates.append(rand_perm)
        
    # Reversed NEH
    if time.time() < deadline - 5:
        neh = _neh_heuristic(processing_times)
        candidates.append(neh[::-1])
        
    # Evaluate candidates
    best_seq = None
    best_ms = float('inf')
    
    for cand in candidates:
        ms = _calc_makespan(processing_times, cand)
        if ms < best_ms:
            best_ms = ms
            best_seq = cand
            
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_ms = _calc_makespan(processing_times, best_seq)
        
    # 2. Local Search
    if time.time() < deadline - 5:
        ls_deadline = deadline - 2
        ls_seq, ls_ms = _local_search(processing_times, best_seq, ls_deadline)
        if ls_ms < best_ms:
            best_seq = ls_seq
            best_ms = ls_ms
            
    return best_seq
# EVOLVE-BLOCK-END
