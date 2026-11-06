"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _calc_makespan(processing_times, order):
    """
    Calculate the makespan for a given job order efficiently.
    processing_times: np.ndarray [n_jobs, n_machines]
    order: list of job indices
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    completion = np.zeros(n_machines, dtype=np.float64)
    
    for job in order:
        pt = processing_times[job]
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(completion[m], prev_completion)
            completion[m] = start_time + pt[m]
            prev_completion = completion[m]
            
    return float(completion[-1])

def _neh_initial(processing_times, rng=None, randomize=False):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    Sort jobs by total processing time descending.
    Insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if rng is None:
        rng = random.Random(42)
        
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    if randomize and n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
                
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _incremental_makespan_after_insert(processing_times, seq, job, pos):
    """
    Efficiently compute makespan of seq with job inserted at position pos.
    Uses prefix completion times to avoid recomputing from scratch.
    """
    n_machines = processing_times.shape[1]
    n = len(seq)
    
    # Compute completion times for prefix seq[:pos]
    prefix_completion = np.zeros(n_machines, dtype=np.float64)
    for i in range(pos):
        pt = processing_times[seq[i]]
        prev_c = 0.0
        for m in range(n_machines):
            start = max(prefix_completion[m], prev_c)
            prefix_completion[m] = start + pt[m]
            prev_c = prefix_completion[m]
    
    # Now process job at position pos
    job_pt = processing_times[job]
    current_completion = np.zeros(n_machines, dtype=np.float64)
    prev_c = 0.0
    for m in range(n_machines):
        start = max(prefix_completion[m], prev_c)
        current_completion[m] = start + job_pt[m]
        prev_c = current_completion[m]
    
    # Then process suffix seq[pos:]
    for i in range(pos, n):
        pt = processing_times[seq[i]]
        prev_c = 0.0
        for m in range(n_machines):
            start = max(current_completion[m], prev_c)
            current_completion[m] = start + pt[m]
            prev_c = current_completion[m]
    
    return float(current_completion[-1])

def _insertion_local_search(processing_times, seq, deadline):
    """
    Local search using insertion moves with incremental evaluation.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_ms = _calc_makespan(processing_times, seq)
    best_ms = current_ms
    best_seq = list(seq)
    
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        best_move_delta = 0
        best_move_i = -1
        best_move_j = -1
        
        for i in range(n_jobs):
            job = best_seq[i]
            seq_reduced = best_seq[:i] + best_seq[i+1:]
            
            for pos in range(n_jobs - 1):
                if pos == i:
                    continue

                ms = _incremental_makespan_after_insert(processing_times, seq_reduced, job, pos)
                delta = ms - best_ms
                
                if delta < best_move_delta - 1e-9:
                    best_move_delta = delta
                    best_move_i = i
                    best_move_j = pos
                    
        if best_move_i != -1:
            job = best_seq[best_move_i]
            seq_reduced = best_seq[:best_move_i] + best_seq[best_move_i+1:]
            best_seq = seq_reduced[:best_move_j] + [job] + seq_reduced[best_move_j:]
            best_ms += best_move_delta
            improved = True
            
        if not improved:
            break
            
    return best_seq, best_ms

def _simulated_annealing(processing_times, seq, deadline, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_seq = list(seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    temperature = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    
    while time.time() < deadline and temperature > 1e-5:
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_ms = _calc_makespan(processing_times, new_seq)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_seq = new_seq
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                
        temperature *= cooling_rate
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: Generate diverse initial solutions using NEH variants
    if time.time() < deadline - 10.0:
        perm1 = _neh_initial(processing_times, rng, randomize=False)
        ms1 = _calc_makespan(processing_times, perm1)
        if ms1 < best_ms:
            best_ms = ms1
            best_perm = perm1
            
    if time.time() < deadline - 10.0:
        perm2 = _neh_initial(processing_times, rng, randomize=True)
        ms2 = _calc_makespan(processing_times, perm2)
        if ms2 < best_ms:
            best_ms = ms2
            best_perm = perm2
            
    if time.time() < deadline - 10.0:
        rng3 = random.Random(12345)
        perm3 = _neh_initial(processing_times, rng3, randomize=True)
        ms3 = _calc_makespan(processing_times, perm3)
        if ms3 < best_ms:
            best_ms = ms3
            best_perm = perm3
            
    # Phase 2: Local Search (Insertion) with incremental evaluation
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=ls_deadline)
        
    # Phase 3: Simulated Annealing for diversification
    if time.time() < deadline - 2.0 and best_perm is not None:
        sa_deadline = deadline - 1.0
        rng_sa = random.Random(99999)
        sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, deadline=sa_deadline, rng=rng_sa)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
            if time.time() < deadline - 0.5:
                best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
