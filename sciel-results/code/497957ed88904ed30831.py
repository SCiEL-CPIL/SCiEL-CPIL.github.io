"""
Flow-shop permutation optimization using fast heuristics, insertion local search,
and simulated annealing with delta evaluation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

TIME_LIMIT = 110.0  # Safety margin from 150s limit

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation. O(N*M).
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    for j in perm:
        curr = 0.0
        for m in range(n_machines):
            if prev[m] > curr:
                curr = prev[m]
            curr += pt[j, m]
            prev[m] = curr
    return prev[-1]

def _makespan_delta_insert(pt, perm, job, pos):
    """
    Compute the delta makespan when inserting job at position pos into perm.
    perm is the list without the job. Returns (new_makespan, delta).
    """
    n_machines = pt.shape[1]
    
    # Compute prefix and suffix completion times
    # prefix[i][m] = completion time at position i (0-indexed) on machine m
    # We need to compute the makespan of the sequence with job inserted at pos.
    
    # First, compute the "flow" of the sequence without the job
    # We need completion times at each position.
    
    # Let's compute prefix: for i in range(len(perm)), prefix[i][m] is C of perm[i] on machine m
    # And suffix: for i in range(len(perm)), suffix[i][m] is the "remaining" time from position i onwards
    
    # Actually, a simpler approach:
    # 1. Compute completion times for the sequence without job: C[i][m]
    # 2. When inserting job at pos, the jobs before pos are unchanged.
    # 3. The job starts at max(C[pos-1][m], start_time_on_machine_m)
    # 4. Jobs after pos are shifted.
    
    # Let's just compute the full makespan for the new permutation.
    # For speed, we can use the fact that we only need to recompute from pos onwards.
    
    new_perm = perm[:pos] + [job] + perm[pos:]
    return _makespan(pt, new_perm)

def _neh_initial(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending.
    Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic.
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _lpt_initial(pt):
    """
    LPT: Sort by longest processing time on first machine.
    """
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(-pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _spt_initial(pt):
    """
    SPT: Sort by shortest processing time on first machine.
    """
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _insertion_local_search(pt, perm, max_iterations=1000, deadline=None):
    """
    First-improvement insertion local search.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    iteration = 0
    while iteration < max_iterations:
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            
            for pos in range(len(without_job) + 1):
                if pos == i:
                    continue
                
                new_perm = without_job[:pos] + [job] + without_job[pos:]
                new_ms = _makespan(pt, new_perm)
                
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
        
        iteration += 1
    
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with Or-Opt neighborhood (size 1, 2, 3).
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    max_block_size = 3
    
    while time.time() < deadline and T > 1e-6:
        block_size = rng.randint(1, max_block_size + 1)
        if n_jobs < block_size:
            block_size = n_jobs
            
        i = rng.randint(0, n_jobs - block_size + 1)
        k = rng.randint(0, n_jobs - block_size + 1)
        
        if k == i or k == i - 1:
            continue
            
        block = current_perm[i:i+block_size]
        
        new_perm = current_perm[:i] + current_perm[i+block_size:]
        new_perm = new_perm[:k] + block + new_perm[k:]
        
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multiple heuristics
    and local search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Random starts
    for seed in range(4):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Phase 1: Insertion local search on each candidate
    # Allocate time: ~10s per candidate
    ls_deadline_per_cand = 10.0
    
    for cand in candidates:
        if time.time() > deadline - 5.0:  # Leave 5s for safety
            break
        cand_deadline = time.time() + ls_deadline_per_cand
        if cand_deadline > deadline - 2.0:
            cand_deadline = deadline - 2.0
            
        perm, ms = _insertion_local_search(pt, cand, max_iterations=500, deadline=cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: If time remains, run Simulated Annealing on the best result
    if time.time() < deadline - 5.0:
        sa_deadline = deadline - 2.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 3: Final local search polish
    if time.time() < deadline - 2.0:
        final_deadline = deadline - 1.0
        perm, ms = _insertion_local_search(pt, best_perm, max_iterations=200, deadline=final_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Ensure we return a list of ints
    return [int(x) for x in best_perm]

# EVOLVE-BLOCK-END
