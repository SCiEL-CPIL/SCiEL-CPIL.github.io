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
    """Local search using insertion moves (remove job, re-insert at best position)."""
    n_jobs = len(seq)
    current_makespan = _calc_makespan(processing_times, seq)
    
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs):
            job = seq[i]
            # Remove job at position i
            seq_reduced = seq[:i] + seq[i+1:]
            best_pos = i
            best_ms = current_makespan
            for pos in range(n_jobs - 1):
                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            if best_ms < current_makespan:
                seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                current_makespan = best_ms
                improved = True
                if deadline is not None and time.time() > deadline:
                    return seq, current_makespan
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
    
    # Stage 2: Local search with insertion moves
    if time.time() < deadline - 10.0:
        best_seq, best_makespan = _insertion_local_search(
            processing_times, best_seq, 
            max_iterations=500, 
            deadline=deadline - 5.0
        )
    
    return best_seq
# EVOLVE-BLOCK-END
