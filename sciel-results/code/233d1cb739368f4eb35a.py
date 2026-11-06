"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    # C[m] = completion time on machine m for the last processed job
    # We simulate the flow shop
    machine_times = np.zeros(n_machines)
    for j in order:
        pt = processing_times[j]
        for m in range(n_machines):
            if m == 0:
                machine_times[m] += pt[m]
            else:
                machine_times[m] = max(machine_times[m], machine_times[m-1]) + pt[m]
    return machine_times[-1]


def _makespan_fast(processing_times, order):
    """Vectorized-ish makespan computation for speed."""
    n_machines = processing_times.shape[1]
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    # Pre-extract processing times in order
    pts = processing_times[order]  # shape (n_jobs, n_machines)
    # Simulate: C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # Use a single array for the previous row
    prev = np.zeros(n_machines)
    for j in range(n_jobs):
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pts[j, 0]
        if n_machines > 1:
            # curr[m] = max(prev[m], curr[m-1]) + pts[j,m]
            # This is a sequential dependency, but we can vectorize the max part
            # Actually the recurrence is: curr[m] = max(prev[m], curr[m-1]) + pts[j, m]
            # This must be done sequentially for m. But n_machines is usually small.
            for m in range(1, n_machines):
                curr[m] = max(prev[m], curr[m-1]) + pts[j, m]
        prev = curr
    return prev[-1]


def _neh_construct(processing_times):
    """NEH heuristic: sort by total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order_indices = list(np.argsort(-total_times))
    
    seq = [order_indices[0]]
    for j in order_indices[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_fast(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _local_search(processing_times, seq, time_limit=110.0):
    """Local search with insertion moves (I1) and swap moves (I2)."""
    n_jobs = len(seq)
    best_ms = _makespan_fast(processing_times, seq)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Try all insertion moves: remove job i, insert at position j
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = seq[i]
            # Remove job from position i
            seq_without = seq[:i] + seq[i+1:]
            for j in range(len(seq_without) + 1):
                if j == i or j == i + 1:
                    continue  # Same or adjacent position, no change
                new_seq = seq_without[:j] + [job] + seq_without[j:]
                ms = _makespan_fast(processing_times, new_seq)
                if ms < best_ms - 1e-9:
                    seq = new_seq
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break  # Restart the outer loop
        
        # If no insertion improved, try swaps
        if not improved and time.time() - start_time < time_limit:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, n_jobs):
                    new_seq = seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    ms = _makespan_fast(processing_times, new_seq)
                    if ms < best_ms - 1e-9:
                        seq = new_seq
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
    
    return seq, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Local search with insertion and swap moves to refine.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Stage 1: NEH construction
    seq = _neh_construct(processing_times)
    best_ms = _makespan_fast(processing_times, seq)
    
    if time.time() > deadline:
        return list(seq)
    
    # Stage 2: Local search
    seq, best_ms = _local_search(processing_times, seq, time_limit=deadline - time.time())
    
    return list(seq)
# EVOLVE-BLOCK-END
