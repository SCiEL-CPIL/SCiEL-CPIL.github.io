"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order in O(n_jobs * n_machines)."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # completion_time[j] tracks the time machine j is free
    completion = np.zeros(n_machines)
    for job_idx in order:
        for j in range(n_machines):
            if j == 0:
                completion[j] += processing_times[job_idx, j]
            else:
                completion[j] = max(completion[j], completion[j-1]) + processing_times[job_idx, j]
    return completion[-1]


def _neh_insertion(processing_times):
    """
    NEH heuristic: sort jobs by total processing time descending,
    then insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence by insertion
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _local_search(processing_times, seq, time_limit=5.0):
    """
    Local search with 2-opt and 3-opt style moves (reversals and shifts).
    Uses a simple first-improvement strategy with a time limit.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq
    
    best_seq = list(seq)
    best_ms = _makespan(processing_times, best_seq)
    start_time = time.time()
    
    # 2-opt: reverse a segment
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                ms = _makespan(processing_times, new_seq)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
    
    # Shift moves: move a job to a different position
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                new_seq = best_seq[:i] + best_seq[i+1:]
                insert_pos = j if j < i else j + 1
                new_seq = new_seq[:insert_pos] + [job] + new_seq[insert_pos:]
                ms = _makespan(processing_times, new_seq)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_seq = new_seq
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
    
    return best_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH insertion
    followed by local search with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Phase 1: NEH insertion heuristic
    best_seq = _neh_insertion(processing_times)
    best_ms = _makespan(processing_times, best_seq)
    
    # Phase 2: Local search with time budget
    remaining = deadline - time.time()
    if remaining > 1.0:
        ls_time = min(remaining * 0.5, 30.0)
        ls_seq = _local_search(processing_times, best_seq, time_limit=ls_time)
        ls_ms = _makespan(processing_times, ls_seq)
        if ls_ms < best_ms:
            best_seq = ls_seq
            best_ms = ls_ms
    
    # Phase 3: Additional local search passes if time remains
    remaining = deadline - time.time()
    if remaining > 2.0:
        ls_time = min(remaining * 0.4, 20.0)
        ls_seq = _local_search(processing_times, best_seq, time_limit=ls_time)
        ls_ms = _makespan(processing_times, ls_seq)
        if ls_ms < best_ms:
            best_seq = ls_seq
            best_ms = ls_ms
    
    # Final check: ensure we're within deadline
    if time.time() > deadline:
        # Return best found so far
        pass
    
    return list(best_seq)
# EVOLVE-BLOCK-END
