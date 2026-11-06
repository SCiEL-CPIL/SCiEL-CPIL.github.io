"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m - 1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    return completion[-1]


def _calc_makespan_fast(processing_times, perm):
    """Vectorized makespan calculation using numpy."""
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[perm].sum())
    
    # Build matrix of processing times for the permutation
    # shape: (n_jobs, n_machines)
    pt = processing_times[np.array(perm)]
    
    # C[0] = cumulative sum along axis 0
    # C[j, m] = max(C[j, m-1], C[j-1, m]) + pt[j, m]
    C = np.empty((len(perm), n_machines), dtype=np.float64)
    C[:, 0] = np.cumsum(pt[:, 0])
    
    for m in range(1, n_machines):
        C[:, m] = np.maximum(C[:, m-1], np.concatenate(([0.0], C[:-1, m]))) + pt[:, m]
    
    return float(C[-1, -1])


def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _reverse_permutation(sequence):
    """Return the reversed sequence."""
    return list(reversed(sequence))


def _shifted_permutation(sequence, shift):
    """Return the sequence rotated by shift positions."""
    n = len(sequence)
    if shift % n == 0:
        return list(sequence)
    return list(sequence[shift % n:]) + list(sequence[:shift % n])


def _swap_pair(sequence, i, j):
    """Return a new sequence with elements at i and j swapped."""
    new_seq = list(sequence)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def _insertion_move(sequence, i, j):
    """
    Remove element at position i and insert it at position j.
    If j > i, adjust for the shift caused by removal.
    """
    new_seq = list(sequence)
    job = new_seq.pop(i)
    if j > i:
        insert_idx = j - 1
    else:
        insert_idx = j
    new_seq.insert(insert_idx, job)
    return new_seq


def _generate_candidate(processing_times, n_jobs, strategy, rng_seed=None):
    """
    Generate a single candidate permutation based on the given strategy.
    strategy: 'neh', 'random', 'reversed_neh', 'shifted_neh', 'random_swap_neh'
    """
    if strategy == 'neh':
        return _neh_heuristic(processing_times)
    elif strategy == 'random':
        return _random_permutation(n_jobs)
    elif strategy == 'reversed_neh':
        return _reverse_permutation(_neh_heuristic(processing_times))
    elif strategy == 'shifted_neh':
        base = _neh_heuristic(processing_times)
        shift = random.randint(1, max(1, n_jobs - 1))
        return _shifted_permutation(base, shift)
    elif strategy == 'random_swap_neh':
        base = _neh_heuristic(processing_times)
        if n_jobs >= 2:
            i, j = random.sample(range(n_jobs), 2)
            return _swap_pair(base, i, j)
        return base
    else:
        return _random_permutation(n_jobs)


def _local_search(processing_times, sequence, deadline):
    """
    Best-improvement local search with both insertion and 2-opt moves.
    Uses vectorized makespan for speed.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_makespan = current_makespan
        best_seq = current_seq
        
        # Try insertion moves
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_seq, current_makespan
            for j in range(n_jobs):
                if j == i:
                    continue
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_seq = new_seq
        
        # Try 2-opt moves (reverse a segment)
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                new_seq = current_seq[:i] + list(reversed(current_seq[i:j+1])) + current_seq[j+1:]
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_seq = new_seq
        
        if best_makespan < current_makespan:
            current_seq = best_seq
            current_makespan = best_makespan
            improved = True
        else:
            improved = False
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with multiple perturbation strategies and restarts.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan_fast(processing_times, sequence)
    
    best_seq = list(sequence)
    best_makespan = _calc_makespan_fast(processing_times, sequence)
    
    # Multiple SA runs with different starting temperatures
    max_runs = 3
    run_time = max(5.0, (deadline - time.time()) / max_runs)
    
    for run in range(max_runs):
        if time.time() >= deadline - 2.0:
            break
        
        current_seq = list(best_seq) if run > 0 else list(sequence)
        current_makespan = _calc_makespan_fast(processing_times, current_seq)
        
        # Vary initial temperature per run
        T = current_makespan * (0.1 + 0.05 * run)
        cooling_rate = 0.9998
        min_T = 1e-8
        run_deadline = min(deadline - 2.0, time.time() + run_time)
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < run_deadline and T > min_T:
            # Choose perturbation type
            r = random.random()
            if r < 0.4:
                # Swap
                i, j = random.sample(range(n_jobs), 2)
                new_seq = _swap_pair(current_seq, i, j)
            elif r < 0.7:
                # Insertion
                i, j = random.sample(range(n_jobs), 2)
                new_seq = _insertion_move(current_seq, i, j)
            else:
                # 2-opt (reverse segment)
                i, j = sorted(random.sample(range(n_jobs), 2))
                new_seq = current_seq[:i] + list(reversed(current_seq[i:j+1])) + current_seq[j+1:]
            
            new_makespan = _calc_makespan_fast(processing_times, new_seq)
            delta = new_makespan - current_makespan
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_seq = new_seq
                current_makespan = new_makespan
                no_improve_count = 0
                
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_seq = list(current_seq)
            else:
                no_improve_count += 1
            
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate diverse initial candidates
    strategies = [
        'neh',
        'random',
        'reversed_neh',
        'shifted_neh',
        'random_swap_neh',
    ]
    
    # Add a few more random variants
    for _ in range(3):
        strategies.append('random')
    
    candidates = []
    for strat in strategies:
        if time.time() >= deadline:
            break
        cand = _generate_candidate(processing_times, n_jobs, strat)
        candidates.append(cand)
    
    # Evaluate all candidates and keep the best
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Phase 1: Local search on the best candidate
    if time.time() < deadline - 10.0:
        ls_seq, ls_makespan = _local_search(processing_times, best_seq, min(deadline - 10.0, time.time() + 25.0))
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 2: Simulated annealing if time permits
    if time.time() < deadline - 15.0:
        sa_deadline = min(deadline - 5.0, time.time() + 50.0)
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, sa_deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 1.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline - 0.5)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
