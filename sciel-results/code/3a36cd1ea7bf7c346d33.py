# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order using the flow-shop recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        c[0] += p_row[0]
        if n_machines > 1:
            c[1:] = np.maximum(c[1:], c[:-1]) + p_row[1:]
    return c[-1]

def _makespan_from_array(processing_times, order_array):
    """Compute makespan from a numpy array of job indices."""
    if len(order_array) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    c = np.zeros(n_machines)
    for job_idx in order_array:
        p_row = processing_times[int(job_idx)]
        c[0] += p_row[0]
        if n_machines > 1:
            c[1:] = np.maximum(c[1:], c[:-1]) + p_row[1:]
    return c[-1]

def _neh_sorting_key(processing_times, variant):
    """Generate a sorting key for NEH variants."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if variant == 'total':
        return processing_times.sum(axis=1)
    elif variant == 'first':
        return processing_times[:, 0]
    elif variant == 'last':
        return processing_times[:, -1]
    elif variant == 'mean':
        return processing_times.mean(axis=1)
    elif variant == 'weighted_first':
        # Weight first machine more
        weights = np.linspace(1.0, 0.5, n_machines)
        return processing_times @ weights
    elif variant == 'weighted_last':
        # Weight last machine more
        weights = np.linspace(0.5, 1.0, n_machines)
        return processing_times @ weights
    elif variant == 'random':
        rng = np.random.RandomState(hash(tuple(processing_times)) % 2**32)
        return rng.random(n_jobs)
    else:
        return processing_times.sum(axis=1)

def _neh_constructive(processing_times, variant='total'):
    """NEH heuristic with different sorting variants."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    key = _neh_sorting_key(processing_times, variant)
    sorted_jobs = list(np.argsort(-key))
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence

def _insertion_local_search(processing_times, order, deadline, max_passes=50):
    """Efficient insertion-based local search with time management."""
    n_jobs = len(order)
    best_order = list(order)
    best_makespan = _makespan(processing_times, best_order)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            
            best_pos = i
            best_ms = best_makespan
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < best_makespan:
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_makespan = best_ms
                improved = True
        
        if not improved:
            break
    
    return best_order, best_makespan

def _swap_local_search(processing_times, order, deadline):
    """Swap-based local search."""
    n_jobs = len(order)
    best_order = list(order)
    best_makespan = _makespan(processing_times, best_order)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() > deadline:
                    break
                # Swap jobs at i and j
                candidate = best_order.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(processing_times, candidate)
                if ms < best_makespan:
                    best_order = candidate
                    best_makespan = ms
                    improved = True
    
    return best_order, best_makespan

def _simulated_annealing(processing_times, order, time_budget, n_iters_per_temp=10):
    """Simulated annealing with insertion moves."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(processing_times, order)
    
    current_order = list(order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    initial_temp = max(1.0, current_makespan * 0.1)
    
    temp = initial_temp
    rng = random.Random(hash(tuple(order)) % 2**32)
    
    while temp > 0.01 and (time.time() - start_time) < time_budget:
        for _ in range(n_iters_per_temp):
            if time.time() > start_time + time_budget:
                break
            
            # Random insertion move
            i = rng.randint(0, n_jobs - 1)
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            valid_positions = [p for p in range(len(remaining) + 1) if p != i and p != i + 1]
            if not valid_positions:
                break
            pos = rng.choice(valid_positions)
            
            neighbor = remaining[:pos] + [job] + remaining[pos:]
            neighbor_makespan = _makespan(processing_times, neighbor)
            delta = neighbor_makespan - current_makespan
            
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current_order = neighbor
                current_makespan = neighbor_makespan
                
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_order = list(current_order)
        
        temp *= 0.95
    
    return best_order, best_makespan

def _tabu_search(processing_times, order, time_budget, tabu_tenure=7):
    """Tabu search with insertion moves."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(processing_times, order)
    
    current_order = list(order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    rng = random.Random(hash(tuple(order)) % 2**32)
    
    # Tabu list: (job, from_pos, to_pos) with expiration
    tabu_list = []
    tabu_iteration = 0
    no_improve = 0
    max_no_improve = 50
    
    while (time.time() - start_time) < time_budget and no_improve < max_no_improve:
        tabu_iteration += 1
        no_improve += 1
        
        # Remove expired tabu moves
        tabu_list = [t for t in tabu_list if t[3] > tabu_iteration]
        
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_move = None
        
        # Sample a subset of moves to keep it fast
        n_samples = min(20, n_jobs * 3)
        for _ in range(n_samples):
            i = rng.randint(0, n_jobs - 1)
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            
            pos = rng.randint(0, len(remaining))
            if pos == i or pos == i + 1:
                continue
            
            # Check tabu
            is_tabu = False
            for t in tabu_list:
                if t[0] == job and t[1] == i and t[2] == pos:
                    is_tabu = True
                    break
            
            candidate = remaining[:pos] + [job] + remaining[pos:]
            ms = _makespan(processing_times, candidate)
            
            # Aspiration: accept tabu move if it improves best
            if not is_tabu or ms < best_makespan:
                if ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor = candidate
                    best_move = (job, i, pos)
        
        if best_neighbor is not None and best_neighbor_ms < current_makespan:
            current_order = best_neighbor
            current_makespan = best_neighbor_ms
            no_improve = 0
            
            # Add to tabu list
            if best_move is not None:
                tabu_list.append((best_move[0], best_move[1], best_move[2], tabu_iteration + tabu_tenure))
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
        elif best_neighbor is not None:
            # Allow worsening move with some probability to escape local optima
            if rng.random() < 0.3:
                current_order = best_neighbor
                current_makespan = best_neighbor_ms
                if best_move is not None:
                    tabu_list.append((best_move[0], best_move[1], best_move[2], tabu_iteration + tabu_tenure))
    
    return best_order, best_makespan

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multi-start NEH, local search, tabu search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    best_order = None
    best_makespan = float('inf')
    
    # Phase 1: Multi-start NEH with different variants
    variants = ['total', 'first', 'last', 'mean', 'weighted_first', 'weighted_last']
    for variant in variants:
        if time.time() > deadline:
            break
        candidate = _neh_constructive(processing_times, variant)
        ms = _makespan(processing_times, candidate)
        if ms < best_makespan:
            best_makespan = ms
            best_order = candidate
    
    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        if time.time() > deadline:
            break
        candidate = list(rng.permutation(n_jobs))
        ms = _makespan(processing_times, candidate)
        if ms < best_makespan:
            best_makespan = ms
            best_order = candidate
    
    if best_order is None:
        best_order = list(range(n_jobs))
        best_makespan = _makespan(processing_times, best_order)
    
    # Phase 2: Insertion local search on best solution
    if time.time() < deadline - 10:
        ls_order, ls_ms = _insertion_local_search(
            processing_times, best_order, deadline - 10, max_passes=30
        )
        if ls_ms < best_makespan:
            best_makespan = ls_ms
            best_order = ls_order
    
    # Phase 3: Tabu search
    if time.time() < deadline - 5:
        tabu_time = min(30.0, deadline - 5 - time.time())
        if tabu_time > 3:
            ts_order, ts_ms = _tabu_search(
                processing_times, best_order, tabu_time, tabu_tenure=7
            )
            if ts_ms < best_makespan:
                best_makespan = ts_ms
                best_order = ts_order
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 3:
        sa_time = min(20.0, deadline - 3 - time.time())
        if sa_time > 3:
            sa_order, sa_ms = _simulated_annealing(
                processing_times, best_order, sa_time, n_iters_per_temp=10
            )
            if sa_ms < best_makespan:
                best_makespan = sa_ms
                best_order = sa_order
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 2:
            fs_order, fs_ms = _insertion_local_search(
                processing_times, best_order, deadline - 2, max_passes=15
            )
            if fs_ms < best_makespan:
                best_makespan = fs_ms
                best_order = fs_order
    
    # Final check: ensure we return a valid permutation
    if sorted(best_order) != list(range(n_jobs)):
        # Fallback to identity if something went wrong
        best_order = list(range(n_jobs))
    
    return best_order
# EVOLVE-BLOCK-END
