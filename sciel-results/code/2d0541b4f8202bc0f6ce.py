"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _makespan_fast(processing_times, order):
    """
    Compute makespan for a given job order using efficient recurrence.
    Uses the standard flow-shop recurrence:
    C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    where C[-1][j] = 0, C[i][-1] = 0
    """
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] holds the completion time of the current job on machine j
    # We process jobs one by one, updating c in-place
    c = [0.0] * n_machines
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _makespan_with_c(processing_times, order):
    """
    Compute makespan and return both the makespan and the final c array.
    This is useful when we need to quickly evaluate insertions.
    """
    if len(order) == 0:
        return 0.0, []
    n_machines = processing_times.shape[1]
    
    c = [0.0] * n_machines
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1], c


def _delta_insertion(processing_times, order, job_idx, pos):
    """
    Compute the change in makespan if we insert job_idx at position pos in order.
    This is an approximation that recomputes the makespan from scratch for accuracy.
    For larger instances, we could optimize this with prefix/suffix completion arrays.
    """
    new_order = order[:pos] + [job_idx] + order[pos:]
    return _makespan_fast(processing_times, new_order)


def _neh_heuristic(processing_times, sort_key='sum'):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for flow shop scheduling.
    Jobs are sorted by a key, then each is inserted at the position
    that minimizes the makespan of the partial sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    if sort_key == 'sum':
        totals = processing_times.sum(axis=1)
    elif sort_key == 'mean':
        totals = processing_times.mean(axis=1)
    elif sort_key == 'min':
        totals = processing_times.min(axis=1)
    elif sort_key == 'max':
        totals = processing_times.max(axis=1)
    elif sort_key == 'range':
        totals = processing_times.max(axis=1) - processing_times.min(axis=1)
    elif sort_key == 'variance':
        totals = processing_times.var(axis=1)
    elif sort_key == 'last':
        totals = processing_times[:, -1]
    elif sort_key == 'first':
        totals = processing_times[:, 0]
    else:
        totals = processing_times.sum(axis=1)
    
    # Sort jobs by the key in descending order
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan_fast(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order


def _local_search_insertion(processing_times, order, time_limit=5.0, max_passes=100):
    """
    First-improvement local search using insertion moves.
    Repeatedly removes each job and reinserts it at the best position.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    start_time = time.time()
    
    for _ in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job = current_order[i]
            # Remove job from position i
            remaining = current_order[:i] + current_order[i+1:]
            
            # Find best insertion position
            best_pos = i
            best_ms = current_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan_fast(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < current_ms:
                current_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
        
        if not improved:
            break
    
    return current_order, current_ms


def _simulated_annealing(processing_times, order, time_limit=10.0):
    """
    Simulated annealing with insertion moves to escape local optima.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    start_time = time.time()
    initial_temp = max(current_ms * 0.02, 1.0)
    cooling_rate = 0.9995
    temp = initial_temp
    
    while time.time() - start_time < time_limit and temp > 0.01:
        # Generate random insertion move
        src = random.randint(0, n_jobs - 1)
        dst = random.randint(0, n_jobs - 1)
        if dst == src or dst == src + 1:
            temp *= cooling_rate
            continue
        
        # Apply move
        job = current_order[src]
        remaining = current_order[:src] + current_order[src+1:]
        # Adjust dst index after removal
        if src < dst:
            dst -= 1
        new_order = remaining[:dst] + [job] + remaining[dst:]
        
        new_ms = _makespan_fast(processing_times, new_order)
        delta = new_ms - current_ms
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        
        temp *= cooling_rate
    
    return best_order, best_ms


def _tabu_search(processing_times, order, time_limit=10.0, tabu_tenure=10):
    """
    Tabu search with insertion moves to escape local optima.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    start_time = time.time()
    tabu_list = {}  # (src, dst) -> iteration when it becomes non-tabu
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Find best non-tabu move
        best_move = None
        best_move_ms = float('inf')
        
        for src in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for dst in range(n_jobs):
                if dst == src or dst == src + 1:
                    continue
                
                # Check if this move is tabu
                key = (src, dst)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                
                job = current_order[src]
                remaining = current_order[:src] + current_order[src+1:]
                if src < dst:
                    dst_adj = dst - 1
                else:
                    dst_adj = dst
                new_order = remaining[:dst_adj] + [job] + remaining[dst_adj:]
                
                new_ms = _makespan_fast(processing_times, new_order)
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (src, dst_adj, job, new_order, new_ms)
        
        if best_move is None:
            break
        
        src, dst_adj, job, new_order, new_ms = best_move
        
        # Check aspiration criterion
        if new_ms < best_ms:
            # Accept even if tabu
            pass
        elif (src, dst_adj) in tabu_list and tabu_list[(src, dst_adj)] > iteration:
            # Skip if tabu and not aspiration
            # Find next best
            continue
        
        # Apply move
        current_order = new_order
        current_ms = new_ms
        
        # Update tabu list
        tabu_list[(src, dst_adj)] = iteration + tabu_tenure
        # Also add reverse move
        reverse_src = dst_adj
        reverse_dst = src if src < dst_adj else src + 1
        tabu_list[(reverse_src, reverse_dst)] = iteration + tabu_tenure
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_order = current_order[:]
        
        # Clean up old tabu entries
        if iteration % 100 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Strategy:
    1. Generate multiple diverse initial solutions using NEH with different sorting keys
    2. Apply local search to each
    3. Use tabu search and simulated annealing on the best to escape local optima
    4. Final local search polish
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: Generate diverse initial solutions using NEH variants
    sort_keys = ['sum', 'mean', 'max', 'min', 'range', 'variance', 'last', 'first']
    candidates = []
    
    for key in sort_keys:
        if time.time() > deadline - 30.0:
            break
        try:
            order = _neh_heuristic(processing_times, sort_key=key)
            ms = _makespan_fast(processing_times, order)
            candidates.append((ms, order))
        except Exception:
            continue
    
    # Also add a random initial solution
    if time.time() < deadline - 30.0:
        random_order = list(range(n_jobs))
        random.shuffle(random_order)
        ms = _makespan_fast(processing_times, random_order)
        candidates.append((ms, random_order))
    
    if not candidates:
        # Fallback: identity permutation
        return list(range(n_jobs))
    
    # Sort by makespan
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top candidates
    ls_time_per_candidate = 3.0
    best_order = candidates[0][1]
    best_ms = candidates[0][0]
    
    for ms, order in candidates[:min(3, len(candidates))]:
        if time.time() > deadline - 40.0:
            break
        ls_order, ls_ms = _local_search_insertion(
            processing_times, order, 
            time_limit=ls_time_per_candidate
        )
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # Phase 3: Tabu search on the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        ts_time = min(remaining_time * 0.4, 40.0)
        ts_order, ts_ms = _tabu_search(
            processing_times, best_order,
            time_limit=ts_time
        )
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_order = ts_order
    
    # Phase 4: Simulated annealing on the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        sa_time = min(remaining_time * 0.5, 30.0)
        sa_order, sa_ms = _simulated_annealing(
            processing_times, best_order,
            time_limit=sa_time
        )
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order
    
    # Phase 5: Final local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        polish_time = min(remaining_time * 0.5, 10.0)
        final_order, final_ms = _local_search_insertion(
            processing_times, best_order,
            time_limit=polish_time
        )
        if final_ms < best_ms:
            best_ms = final_ms
            best_order = final_order
    
    return best_order
# EVOLVE-BLOCK-END
