"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        completion[0] += row[0]
        for m in range(1, n_machines):
            c_prev = completion[m - 1]
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + row[m]
    return completion[-1]


def _calc_makespan_state(processing_times, perm):
    """Calculate makespan and return full completion state."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        completion[0] += row[0]
        for m in range(1, n_machines):
            c_prev = completion[m - 1]
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + row[m]
    return completion[-1], completion


def _eval_insertion(processing_times, seq, job, pos, prefix_state):
    """
    Evaluate inserting job at position pos in seq.
    prefix_state: completion times after processing seq[:pos].
    Returns makespan.
    """
    n_machines = processing_times.shape[1]
    completion = list(prefix_state)
    row = processing_times[job]
    # Process the job
    completion[0] += row[0]
    for m in range(1, n_machines):
        c_prev = completion[m - 1]
        c_curr = completion[m]
        if c_curr < c_prev:
            c_curr = c_prev
        completion[m] = c_curr + row[m]
    # Process remaining jobs
    for idx in range(pos, len(seq)):
        r = processing_times[seq[idx]]
        completion[0] += r[0]
        for m in range(1, n_machines):
            c_prev = completion[m - 1]
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + r[m]
    return completion[-1]


def _get_prefix_state(processing_times, seq, pos):
    """Get completion state after processing seq[:pos]."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    for idx in range(pos):
        row = processing_times[seq[idx]]
        completion[0] += row[0]
        for m in range(1, n_machines):
            c_prev = completion[m - 1]
            c_curr = completion[m]
            if c_curr < c_prev:
                c_curr = c_prev
            completion[m] = c_curr + row[m]
    return completion


def _neh_heuristic(processing_times):
    """NEH heuristic with efficient insertion evaluation."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Compute prefix states incrementally
        prefix_states = []
        completion = [0.0] * n_machines
        prefix_states.append(completion[:])
        for idx in range(len(sequence)):
            row = processing_times[sequence[idx]]
            completion[0] += row[0]
            for m in range(1, n_machines):
                c_prev = completion[m - 1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
            prefix_states.append(completion[:])
        
        for pos in range(len(sequence) + 1):
            ms = _eval_insertion(processing_times, sequence, job, pos, prefix_states[pos])
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search_insertion(processing_times, sequence, deadline):
    """
    Insertion local search: try moving a job to every other position.
    Uses incremental evaluation to speed up.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            # Precompute prefix state up to i
            prefix_state = _get_prefix_state(processing_times, current_seq, i)
            
            best_j = -1
            best_ms = current_makespan
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Remove job at i, insert at j
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                # Build new sequence
                new_seq = current_seq[:i] + current_seq[i+1:]
                new_seq.insert(insert_idx, job)
                
                # Determine prefix length for incremental eval
                if j > i:
                    # Prefix is first i elements (same as before removal)
                    # But actually after removal, positions shift
                    # Simpler: just compute full makespan for correctness
                    new_ms = _calc_makespan(processing_times, new_seq)
                else:
                    new_ms = _calc_makespan(processing_times, new_seq)
                
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_j = j
            
            if best_j >= 0:
                # Apply the best move
                new_seq = current_seq[:i] + current_seq[i+1:]
                if best_j > i:
                    insert_idx = best_j - 1
                else:
                    insert_idx = best_j
                new_seq.insert(insert_idx, job)
                current_seq = new_seq
                current_makespan = best_ms
                improved = True
                break
    
    return current_seq, current_makespan


def _local_search_2opt(processing_times, sequence, deadline):
    """
    2-opt local search: reverse a segment of the permutation.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() >= deadline:
                    break
                
                # Reverse segment [i, j]
                new_seq = current_seq[:]
                new_seq[i:j+1] = reversed(new_seq[i:j+1])
                
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                if new_makespan < current_makespan:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    improved = True
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with adaptive temperature and insertion moves.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.15
    min_T = 1e-4
    cooling_rate = 0.998
    
    steps = 0
    max_steps = int((deadline - time.time()) * 500)  # ~500 iterations per second
    
    while time.time() < deadline and T > min_T and steps < max_steps:
        steps += 1
        
        # Choose move type: 70% insertion, 30% swap
        if n_jobs < 2:
            break
        
        if random.random() < 0.7:
            # Insertion move: remove random job, insert at random position
            i = random.randint(0, n_jobs - 1)
            job = current_seq[i]
            new_seq = current_seq[:i] + current_seq[i+1:]
            j = random.randint(0, len(new_seq) - 1)
            new_seq.insert(j, job)
        else:
            # Swap move
            i, j = random.sample(range(n_jobs), 2)
            new_seq = current_seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Phase 1: NEH construction
    initial_seq = _neh_heuristic(processing_times)
    best_seq, best_makespan = _local_search_insertion(processing_times, initial_seq, deadline)
    
    # Phase 2: 2-opt refinement
    if time.time() < deadline - 10.0:
        seq2, ms2 = _local_search_2opt(processing_times, best_seq, deadline)
        if ms2 < best_makespan:
            best_seq, best_makespan = seq2, ms2
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_seq, sa_ms = _simulated_annealing(processing_times, best_seq, deadline)
        if sa_ms < best_makespan:
            best_seq, best_makespan = sa_seq, sa_ms
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_ms = _local_search_insertion(processing_times, best_seq, deadline)
        if final_ms < best_makespan:
            best_seq, best_makespan = final_seq, final_ms
    
    # Phase 5: One more 2-opt pass if time allows
    if time.time() < deadline - 1.0:
        opt_seq, opt_ms = _local_search_2opt(processing_times, best_seq, deadline)
        if opt_ms < best_makespan:
            best_seq, best_makespan = opt_seq, opt_ms
    
    return best_seq


# EVOLVE-BLOCK-END
