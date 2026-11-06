"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(pt, seq):
    """Compute makespan for a given sequence."""
    n_machines = pt.shape[1]
    cum = np.zeros(n_machines)
    for j in seq:
        row = pt[j]
        for k in range(n_machines):
            cum[k] += row[k]
    return cum[-1]


def _neh_initial(pt):
    """NEH heuristic: sort by total processing time descending, insert each job in best position."""
    n_jobs = pt.shape[0]
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(pt, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, j)
    return seq


def _local_search(pt, seq, deadline):
    """
    Multi-pass best-improvement local search over adjacent and 2-swap neighborhoods.
    Uses a deadline guard to stop before timeout.
    """
    n_jobs = len(seq)
    n_machines = pt.shape[1]
    current_ms = _makespan(pt, seq)
    best_seq = list(seq)
    best_ms = current_ms
    improved = True
    passes = 0
    
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Adjacent swap neighborhood
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                return best_seq
            trial = list(seq)
            trial[i], trial[i+1] = trial[i+1], trial[i]
            ms = _makespan(pt, trial)
            if ms < current_ms - 1e-9:
                seq = trial
                current_ms = ms
                improved = True
                # Restart scan from beginning since sequence changed
                break
        
        # If no adjacent swap improved, try 2-swap (non-adjacent)
        if not improved:
            for i in range(n_jobs - 2):
                if time.time() >= deadline:
                    return best_seq
                for j in range(i + 2, n_jobs):
                    if time.time() >= deadline:
                        return best_seq
                    trial = list(seq)
                    trial[i], trial[j] = trial[j], trial[i]
                    ms = _makespan(pt, trial)
                    if ms < current_ms - 1e-9:
                        seq = trial
                        current_ms = ms
                        improved = True
                        break
                if improved:
                    break
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = list(seq)
    
    return best_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Multi-pass local search with adjacent and 2-swap moves.
    3. Multiple restarts with perturbation if time permits.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initial solution via NEH
    seq = _neh_initial(processing_times)
    best_seq = list(seq)
    best_ms = _makespan(processing_times, seq)
    
    # Local search on the initial solution
    if time.time() < deadline:
        seq = _local_search(processing_times, seq, deadline)
        ms = _makespan(processing_times, seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
    
    # Restart with perturbation if time permits
    while time.time() < deadline - 5.0:
        # Perturb the best sequence
        perturbed = list(best_seq)
        n_perturb = max(2, n_jobs // 4)
        for _ in range(n_perturb):
            i = np.random.randint(0, n_jobs - 1)
            j = i + 1 + np.random.randint(0, max(1, n_jobs - i - 1))
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Local search from perturbed solution
        if time.time() < deadline:
            candidate = _local_search(processing_times, perturbed, deadline)
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_seq = list(candidate)
    
    return best_seq
# EVOLVE-BLOCK-END
