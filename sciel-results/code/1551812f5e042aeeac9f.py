"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, order):
    """Calculate the makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # C[m] = completion time on machine m for the current job
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job, m]
            else:
                completion[m] = max(completion[m - 1], completion[m]) + processing_times[job, m]
    return completion[-1]


def _compute_prefix_completions(processing_times, seq):
    """Compute prefix completion times: prefix[i][m] = completion time on machine m after processing jobs 0..i-1."""
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    prefix = np.zeros((n_jobs + 1, n_machines))
    for i in range(n_jobs):
        job = seq[i]
        for m in range(n_machines):
            if m == 0:
                prefix[i + 1, m] = prefix[i, m] + processing_times[job, m]
            else:
                prefix[i + 1, m] = max(prefix[i + 1, m - 1], prefix[i, m]) + processing_times[job, m]
    return prefix


def _compute_suffix_completions(processing_times, seq):
    """Compute suffix completion times: suffix[i][m] = additional time on machine m when processing jobs i..n-1 starting from time 0."""
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    suffix = np.zeros((n_jobs + 1, n_machines))
    for i in range(n_jobs - 1, -1, -1):
        job = seq[i]
        for m in range(n_machines - 1, -1, -1):
            if m == n_machines - 1:
                suffix[i, m] = suffix[i + 1, m] + processing_times[job, m]
            else:
                suffix[i, m] = max(suffix[i, m + 1], suffix[i + 1, m]) + processing_times[job, m]
    return suffix


def _eval_insert_incremental(processing_times, seq, prefix, suffix, job, pos):
    """
    Evaluate the makespan if job is inserted at position pos in seq.
    Uses precomputed prefix and suffix arrays for O(n_machines) evaluation.
    """
    n_machines = processing_times.shape[1]
    
    # Time when machine m is free before position pos
    before = prefix[pos].copy()
    # Additional time from jobs after pos
    after = suffix[pos].copy()
    
    # Simulate inserting job at position pos
    # The job starts at max(before[m-1], before[m]) on machine m
    # Then the remaining jobs shift
    new_completion = np.zeros(n_machines)
    
    # Process the inserted job
    for m in range(n_machines):
        if m == 0:
            new_completion[m] = before[m] + processing_times[job, m]
        else:
            new_completion[m] = max(before[m], new_completion[m - 1]) + processing_times[job, m]
    
    # Now process the remaining jobs (originally at positions pos, pos+1, ...)
    # Their start times are affected by the new completion times
    for i in range(pos, len(seq)):
        j = seq[i]
        for m in range(n_machines):
            if m == 0:
                new_completion[m] += processing_times[j, m]
            else:
                new_completion[m] = max(new_completion[m - 1], new_completion[m]) + processing_times[j, m]
    
    return new_completion[-1]


def _best_insert_position(processing_times, seq, prefix, suffix, job):
    """Find the best position to insert job into seq using incremental evaluation."""
    n_jobs = len(seq)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_jobs + 1):
        ms = _eval_insert_incremental(processing_times, seq, prefix, suffix, job, pos)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return best_pos, best_ms


def _neh_initial(processing_times):
    """NEH (Nawaz, Enscore, Ham) heuristic for initial construction."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _insertion_local_search(processing_times, seq, max_iterations=1000, deadline=None):
    """
    Local search using insertion moves with incremental evaluation.
    For each job, remove it and find the best reinsertion position.
    """
    n_jobs = len(seq)
    if n_jobs == 0:
        return seq, 0.0
    
    current_makespan = _calc_makespan(processing_times, seq)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                return seq, current_makespan
            
            job = seq[i]
            # Remove job at position i
            seq_reduced = seq[:i] + seq[i+1:]
            
            # Compute prefix and suffix for the reduced sequence
            prefix = _compute_prefix_completions(processing_times, seq_reduced)
            suffix = _compute_suffix_completions(processing_times, seq_reduced)
            
            # Find best position to reinsert
            best_pos, best_ms = _best_insert_position(processing_times, seq_reduced, prefix, suffix, job)
            
            if best_ms < current_makespan - 1e-9:
                seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                current_makespan = best_ms
                improved = True
        
        if not improved:
            break
    
    return seq, current_makespan


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Stage 1: NEH heuristic
    best_seq = _neh_initial(processing_times)
    best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Stage 2: Local search with insertion moves using incremental evaluation
    if time.time() < deadline - 10.0:
        best_seq, best_makespan = _insertion_local_search(
            processing_times, best_seq, 
            max_iterations=2000, 
            deadline=deadline - 5.0
        )
    
    return best_seq
# EVOLVE-BLOCK-END
