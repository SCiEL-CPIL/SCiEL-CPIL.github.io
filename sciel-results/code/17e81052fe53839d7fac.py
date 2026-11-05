"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation on flow-shop."""
    n = len(perm)
    m = p.shape[1]
    # completion time matrix: C[i][j] = completion of job i on machine j
    # We compute incrementally
    c = 0.0
    max_c = 0.0
    # For each job in order, track cumulative time on each machine
    # C[job, machine] = max(C[prev_job, machine], C[job, prev_machine]) + p[job, machine]
    # More efficient: keep an array of completion times per machine
    machine_end = np.zeros(m)
    for j in perm:
        row = p[j]
        for k in range(m):
            if k == 0:
                machine_end[0] += row[0]
            else:
                machine_end[k] = max(machine_end[k - 1], machine_end[k]) + row[k]
        max_c = max(max_c, machine_end[-1])
    return max_c


def _neh_order(p):
    """Construct initial ordering using NEH heuristic."""
    n = p.shape[0]
    m = p.shape[1]
    totals = p.sum(axis=1)
    # Sort by total processing time descending
    order = np.argsort(-totals)
    
    # NEH insertion
    perm = [order[0]]
    for idx in order[1:]:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [idx] + perm[pos:]
            ms = _makespan(trial, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [idx] + perm[best_pos:]
    return perm


def _two_opt(perm, p, max_iter=500, time_limit=10.0):
    """Improve permutation using 2-opt moves."""
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    start = time.time()
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start > time_limit:
            break
        improved = False
        iter_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                trial = list(best_perm)
                # Reverse segment between i and j
                trial[i:j+1] = reversed(trial[i:j+1])
                ms = _makespan(trial, p)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = trial
                    improved = True
    return best_perm


def _or_opt(perm, p, max_iter=200, time_limit=8.0):
    """Improve permutation using OR-opt moves (relocate a block of 1-2 jobs)."""
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, p)
    start = time.time()
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start > time_limit:
            break
        improved = False
        iter_count += 1
        for i in range(n):
            for block_size in [1, 2]:
                if i + block_size > n:
                    continue
                block = best_perm[i:i+block_size]
                for j in range(n - block_size + 1):
                    if j == i:
                        continue
                    # Skip if adjacent
                    if j == i + 1 or (j + block_size == i):
                        continue
                    trial = best_perm[:i] + best_perm[i+block_size:j] + block + best_perm[j:i] + best_perm[i+block_size:j+block_size] if j < i else best_perm[:j] + block + best_perm[j:i] + best_perm[i+block_size:]
                    # Let me redo this more carefully
                    # Remove block from position i
                    rest = best_perm[:i] + best_perm[i+block_size:]
                    # Insert at position j in rest
                    # But j needs adjustment if j > i
                    insert_pos = j if j < i else j - block_size
                    if insert_pos < 0:
                        insert_pos = 0
                    if insert_pos > len(rest):
                        insert_pos = len(rest)
                    trial = rest[:insert_pos] + block + rest[insert_pos:]
                    if len(trial) != n:
                        continue
                    ms = _makespan(trial, p)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = trial
                        improved = True
    return best_perm


def order_jobs(processing_times):
    """Order jobs to minimize makespan on a flow-shop."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit, keep 10s buffer
    
    p = np.asarray(processing_times, dtype=np.float64)
    n, m = p.shape
    
    if n <= 1:
        return list(range(n))
    
    # Generate initial solution using NEH
    best_perm = _neh_order(p)
    best_ms = _makespan(best_perm, p)
    
    # Check time
    if time.time() > deadline:
        return best_perm
    
    # Local search: 2-opt
    perm = _two_opt(best_perm, p, max_iter=300, time_limit=20.0)
    ms = _makespan(perm, p)
    if ms < best_ms:
        best_perm = perm
        best_ms = ms
    
    if time.time() > deadline:
        return best_perm
    
    # Local search: OR-opt
    perm = _or_opt(best_perm, p, max_iter=150, time_limit=15.0)
    ms = _makespan(perm, p)
    if ms < best_ms:
        best_perm = perm
        best_ms = ms
    
    if time.time() > deadline:
        return best_perm
    
    # Additional 2-opt passes if time remains
    remaining = deadline - time.time()
    if remaining > 5.0:
        perm = _two_opt(best_perm, p, max_iter=200, time_limit=remaining - 2.0)
        ms = _makespan(perm, p)
        if ms < best_ms:
            best_perm = perm
            best_ms = ms
    
    return best_perm
# EVOLVE-BLOCK-END
