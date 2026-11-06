"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_seq(pt, seq):
    """Compute makespan for a given sequence of job indices.
    pt: (n_jobs, n_machines) array
    seq: list of job indices in order
    """
    n_machines = pt.shape[1]
    # completion[i] = completion time of job i on each machine
    # We track the completion time on the last machine for each job to get makespan
    # Actually we need the completion time on machine m for the last job
    # Let's use a 1D array for the current machine completion times
    comp = np.zeros(n_machines)
    for job in seq:
        t = 0
        for m in range(n_machines):
            t = max(t, comp[m]) + pt[job, m]
            comp[m] = t
    return comp[-1]


def _makespan_fast(pt, seq):
    """Faster makespan using list operations."""
    n_machines = pt.shape[1]
    comp = [0.0] * n_machines
    for job in seq:
        t = 0.0
        row = pt[job]
        for m in range(n_machines):
            if comp[m] > t:
                t = comp[m]
            t += row[m]
            comp[m] = t
    return comp[-1]


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Local search with insertion moves and swap moves.
    3. Deadline guard to ensure completion within time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Convert to list of lists for faster access in inner loops
    pt = processing_times.tolist()
    
    # --- Stage 1: NEH Heuristic ---
    total_times = [sum(pt[j]) for j in range(n_jobs)]
    order = sorted(range(n_jobs), key=lambda j: total_times[j], reverse=True)
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting j at each position
        for pos in range(len(seq) + 1):
            # Create new sequence
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_fast(pt, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    best_seq = seq[:]
    best_ms = _makespan_fast(pt, best_seq)
    
    # --- Stage 2: Local Search ---
    # Use insertion moves and swap moves
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        improved = False
        
        # Try all insertion moves
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Remove job from position i, insert at position j
                # Need to be careful with index adjustment
                new_seq = best_seq[:i] + best_seq[i+1:]
                if j > i:
                    j -= 1
                if j >= len(new_seq):
                    j = len(new_seq) - 1
                new_seq = new_seq[:j] + [job] + new_seq[j:]
                ms = _makespan_fast(pt, new_seq)
                if ms < best_ms - 1e-9:
                    best_seq = new_seq
                    best_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try all swap moves
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                new_seq = best_seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                ms = _makespan_fast(pt, new_seq)
                if ms < best_ms - 1e-9:
                    best_seq = new_seq
                    best_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_seq
# EVOLVE-BLOCK-END
