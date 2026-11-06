"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(seq, processing_times):
    """Compute makespan for a given job sequence."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in seq:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job, m]
            else:
                completion[m] = max(completion[m], completion[m - 1]) + processing_times[job, m]
    return completion[-1]


def _makespan_fast(seq, processing_times):
    """Vectorized makespan computation using cumulative sums where possible."""
    n_machines = processing_times.shape[1]
    # For a flow shop, the completion time on machine m for job at position i
    # is the max of (completion on machine m-1 for same job, completion on machine m for previous job) + processing time
    # This is inherently sequential, but we can use numpy for the inner loop
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in seq:
        pt = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[m] += pt[m]
            else:
                completion[m] = max(completion[m], completion[m - 1]) + pt[m]
    return completion[-1]


def _neh_initial(processing_times):
    """NEH heuristic: sort by total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_fast(new_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _insertion_local_search(seq, processing_times, max_iter=100, time_limit=None):
    """Local search using insertion moves (I1)."""
    n_jobs = len(seq)
    best_seq = list(seq)
    best_ms = _makespan_fast(best_seq, processing_times)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        improved = False
        for i in range(n_jobs):
            job = best_seq[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Move job from position i to position j
                new_seq = best_seq[:i] + best_seq[i+1:]
                if j > i:
                    j -= 1
                new_seq = new_seq[:j] + [job] + new_seq[j:]
                ms = _makespan_fast(new_seq, processing_times)
                if ms < best_ms - 1e-9:
                    best_seq = new_seq
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_seq, best_ms


def _perturb(seq, strength=2):
    """Perturb the sequence by moving 'strength' random jobs to random positions."""
    n_jobs = len(seq)
    new_seq = list(seq)
    for _ in range(min(strength, n_jobs - 1)):
        i = np.random.randint(0, n_jobs)
        j = np.random.randint(0, n_jobs)
        if i == j:
            continue
        job = new_seq.pop(i)
        new_seq.insert(j, job)
    return new_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Iterated local search with insertion moves and perturbation.
    
    Returns a list of job indices representing the optimal permutation found.
    """
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Stage 1: NEH initial solution
    best_seq = _neh_initial(processing_times)
    best_ms = _makespan_fast(best_seq, processing_times)
    
    # Stage 2: Local search to polish
    best_seq, best_ms = _insertion_local_search(best_seq, processing_times, max_iter=50)
    
    # Stage 3: Iterated local search with perturbation
    rng = np.random.default_rng(42)
    no_improve_count = 0
    
    while time.time() < deadline:
        # Perturb
        perturbed_seq = _perturb(best_seq, strength=2 + int(2 * rng.random()))
        
        # Local search on perturbed solution
        candidate_seq, candidate_ms = _insertion_local_search(
            perturbed_seq, processing_times, max_iter=30, time_limit=deadline
        )
        
        if candidate_ms < best_ms - 1e-9:
            best_seq = candidate_seq
            best_ms = candidate_ms
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Occasionally accept worse solutions to escape local optima
            if no_improve_count > 10 and rng.random() < 0.1:
                best_seq = candidate_seq
                best_ms = candidate_ms
                no_improve_count = 0
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    return list(best_seq)
# EVOLVE-BLOCK-END
