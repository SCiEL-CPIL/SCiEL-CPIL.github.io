"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_order(processing_times, order):
    """Compute makespan for a given job permutation."""
    n_machines = processing_times.shape[1]
    # completion[m] = completion time on machine m for the last job processed
    completion = np.zeros(n_machines, dtype=np.float64)
    for job_idx in order:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m - 1] if m > 0 else 0) + processing_times[job_idx, m]
    return completion[-1]


def _neh_initialization(processing_times, seed=None):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs = processing_times.shape[0]
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    totals = processing_times.sum(axis=1)
    # Sort jobs by total processing time descending
    sorted_jobs = np.argsort(-totals)
    
    # Build permutation by inserting each job at best position
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = np.inf
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_from_order(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _improve_with_insertion(processing_times, perm, max_passes=20, deadline=None):
    """Insertion-based local search: move each job to its best position."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_makespan = _makespan_from_order(processing_times, perm)
    improved = True
    
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(n_jobs):
            job = perm[i]
            # Try inserting job at every other position
            for j in range(n_jobs):
                if i == j:
                    continue
                # Remove job from position i
                new_perm = perm[:i] + perm[i+1:]
                # Insert at position j (adjust if j > i)
                insert_pos = j if j < i else j
                new_perm = new_perm[:insert_pos] + [job] + new_perm[insert_pos:]
                
                ms = _makespan_from_order(processing_times, new_perm)
                if ms < current_makespan:
                    current_makespan = ms
                    perm = new_perm
                    improved = True
                    break
            if deadline is not None and time.time() >= deadline:
                break
    
    return perm


def _two_opt(processing_times, perm, max_passes=10, deadline=None):
    """2-opt style local search for permutation (reverse segments)."""
    n_jobs = len(perm)
    if n_jobs <= 2:
        return perm
    
    current_makespan = _makespan_from_order(processing_times, perm)
    improved = True
    
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                # Reverse segment from i to j
                new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                ms = _makespan_from_order(processing_times, new_perm)
                if ms < current_makespan:
                    current_makespan = ms
                    perm = new_perm
                    improved = True
                    break
            if improved:
                break
    
    return perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH initialization
    with multiple restarts and insertion-based local search.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Start with standard NEH
    best_perm = _neh_initialization(processing_times)
    best_makespan = _makespan_from_order(processing_times, best_perm)
    
    # Improve with insertion local search
    best_perm = _improve_with_insertion(processing_times, best_perm, max_passes=30, deadline=deadline)
    best_makespan = _makespan_from_order(processing_times, best_perm)
    
    # Try multiple diverse restarts
    np.random.seed(42)
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Create perturbed initial ordering
        totals = processing_times.sum(axis=1)
        sorted_jobs = np.argsort(-totals)
        
        # Perturb the order
        if r % 3 == 0:
            # Reverse some portion
            cut = np.random.randint(1, n_jobs)
            perturbed = list(sorted_jobs[:cut]) + list(sorted_jobs[cut:][::-1])
        elif r % 3 == 1:
            # Random swap of a few adjacent pairs
            perturbed = list(sorted_jobs)
            for _ in range(min(3, n_jobs // 2)):
                idx = np.random.randint(0, n_jobs - 1)
                perturbed[idx], perturbed[idx + 1] = perturbed[idx + 1], perturbed[idx]
        else:
            # Random shuffle of a subset
            perturbed = list(sorted_jobs)
            subset_size = max(2, n_jobs // 4)
            indices = np.random.choice(n_jobs, size=min(subset_size, n_jobs), replace=False)
            subset = [perturbed[i] for i in indices]
            np.random.shuffle(subset)
            for k, idx in enumerate(indices):
                perturbed[idx] = subset[k]
        
        # Build NEH from perturbed order
        perm = []
        for job in perturbed:
            best_pos = 0
            best_ms = np.inf
            for pos in range(len(perm) + 1):
                candidate = perm[:pos] + [job] + perm[pos:]
                ms = _makespan_from_order(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            perm = perm[:best_pos] + [job] + perm[best_pos:]
        
        # Local search
        perm = _improve_with_insertion(processing_times, perm, max_passes=15, deadline=deadline)
        ms = _makespan_from_order(processing_times, perm)
        
        if ms < best_makespan:
            best_makespan = ms
            best_perm = perm
    
    # Final polish with 2-opt
    if time.time() < deadline:
        best_perm = _two_opt(processing_times, best_perm, max_passes=5, deadline=deadline)
    
    return best_perm
# EVOLVE-BLOCK-END
