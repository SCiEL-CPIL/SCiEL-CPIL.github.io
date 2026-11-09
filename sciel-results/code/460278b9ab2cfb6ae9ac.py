"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation on a flow shop."""
    n_machines = processing_times.shape[1]
    # completion[i] = completion time of the current job on machine i
    # We track completion times per machine as we process jobs in order
    machine_finish = np.zeros(n_machines)
    for job in perm:
        prev_job_finish = 0.0
        for m in range(n_machines):
            # Job can start on machine m only after:
            # 1. The previous job finished on machine m (machine_finish[m])
            # 2. The current job finished on machine m-1 (prev_job_finish)
            start = max(machine_finish[m], prev_job_finish)
            prev_job_finish = start + processing_times[job, m]
            machine_finish[m] = prev_job_finish
    return machine_finish[-1]


def _neh_variant(processing_times, sort_key):
    """Generate a NEH-style ordering using the given sort key."""
    n_jobs = processing_times.shape[0]
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Create jobs sorted by the given key (descending)
    if sort_key == 'total':
        keys = processing_times.sum(axis=1)
    elif sort_key == 'first_last':
        # NEH original: sort by sum of first and last machine times
        keys = processing_times[:, 0] + processing_times[:, -1]
    elif sort_key == 'avg':
        keys = processing_times.mean(axis=1)
    elif sort_key == 'max':
        keys = processing_times.max(axis=1)
    else:
        keys = processing_times.sum(axis=1)
    
    sorted_jobs = np.argsort(-keys)
    
    # Insert each job one by one into the best position
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting at each position
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _or_opt(processing_times, perm, max_iterations=100):
    """Or-opt style local search: remove a segment of 1-3 jobs and reinsert."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        # Try removing segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            for i in range(n_jobs - seg_len + 1):
                # Extract segment
                segment = current_perm[i:i+seg_len]
                remaining = current_perm[:i] + current_perm[i+seg_len:]
                
                # Try reinserting at each position in remaining
                for j in range(len(remaining) - seg_len + 1):
                    if i == j and seg_len == 1:
                        continue  # Skip no-op
                    # Avoid inserting back at the same relative position
                    if j >= i and j < i + seg_len:
                        continue
                    
                    trial = remaining[:j] + segment + remaining[j:]
                    ms = _makespan(trial, processing_times)
                    if ms < current_ms - 1e-9:
                        current_perm = trial
                        current_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_perm


def _swap_local_search(processing_times, perm, max_iterations=50):
    """Adjacent swap local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs - 1):
            # Swap jobs i and i+1
            trial = current_perm[:]
            trial[i], trial[i+1] = trial[i+1], trial[i]
            ms = _makespan(trial, processing_times)
            if ms < current_ms - 1e-9:
                current_perm = trial
                current_ms = ms
                improved = True
                # Restart from beginning after improvement
                break
        
        if not improved:
            break
    
    return current_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to leave 10s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate multiple diverse initial solutions using different NEH variants
    variants = ['total', 'first_last', 'avg', 'max']
    
    best_perm = None
    best_ms = float('inf')
    
    for variant in variants:
        if time.time() > deadline:
            break
        perm = _neh_variant(processing_times, variant)
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    if best_perm is None:
        return list(range(n_jobs))
    
    # Local search refinement: Or-opt first, then adjacent swaps
    # Use multiple rounds if time permits
    while time.time() < deadline:
        # Or-opt local search
        improved_perm = _or_opt(processing_times, best_perm, max_iterations=50)
        improved_ms = _makespan(improved_perm, processing_times)
        
        if improved_ms < best_ms - 1e-9:
            best_perm = improved_perm
            best_ms = improved_ms
        
        # Adjacent swap local search
        swapped_perm = _swap_local_search(processing_times, best_perm, max_iterations=30)
        swapped_ms = _makespan(swapped_perm, processing_times)
        
        if swapped_ms < best_ms - 1e-9:
            best_perm = swapped_perm
            best_ms = swapped_ms
        else:
            # No further improvement from either method
            break
    
    return best_perm
# EVOLVE-BLOCK-END
