"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using vectorized numpy ops."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Extract the submatrix for this permutation
    pt_matrix = processing_times[perm]  # shape (n_jobs, n_machines)
    
    # For flow shop: C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    # We can compute this iteratively but use numpy for the inner loop
    c_prev = np.zeros(n_machines, dtype=np.float64)  # C[j-1][m] for all m
    for j in range(n_jobs):
        p = pt_matrix[j]  # P[j][m]
        c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + p
        c_prev = c_curr
    
    return float(c_prev[-1])


def _makespan_from_state(perm, processing_times, start_idx, machine_state):
    """
    Compute makespan from a given starting position with known machine completion times.
    machine_state: array of completion times on each machine before position start_idx.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    c_prev = np.array(machine_state, dtype=np.float64)
    for j in range(start_idx, n_jobs):
        p = processing_times[perm[j]]
        c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + p
        c_prev = c_curr
    
    return float(c_prev[-1])


def _get_machine_state(perm, processing_times, up_to):
    """Get machine completion times after processing jobs 0..up_to-1 in perm."""
    n_machines = processing_times.shape[1]
    c_prev = np.zeros(n_machines, dtype=np.float64)
    for j in range(up_to):
        p = processing_times[perm[j]]
        c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + p
        c_prev = c_curr
    return c_prev


def _evaluate_swap(perm, processing_times, i, j, machine_states):
    """
    Evaluate swap of positions i and j using precomputed machine states.
    machine_states[k] = machine completion times after processing positions 0..k-1.
    Returns the new makespan.
    """
    n_jobs = len(perm)
    start = min(i, j)
    
    # Get state at start position
    c_prev = np.array(machine_states[start], dtype=np.float64)
    
    for k in range(start, n_jobs):
        if k == i:
            job = perm[j]
        elif k == j:
            job = perm[i]
        else:
            job = perm[k]
        p = processing_times[job]
        c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + p
        c_prev = c_curr
    
    return float(c_prev[-1])


def _evaluate_insert(perm, processing_times, i, j, machine_states):
    """
    Evaluate moving job at position i to position j (insertion move).
    If i < j: remove from i, insert at j (shifts i+1..j left by 1)
    If i > j: remove from i, insert at j (shifts j..i-1 right by 1)
    machine_states[k] = machine completion times after processing positions 0..k-1.
    Returns the new makespan.
    """
    n_jobs = len(perm)
    
    if i < j:
        start = i
        # Build the new sequence from position i onward:
        # Original: perm[i], perm[i+1], ..., perm[j], ...
        # New: perm[i+1], perm[i+2], ..., perm[j], perm[i], perm[j+1], ...
        # i.e., perm[i] moves from position i to position j
    else:
        start = j
        # Original: ..., perm[j], ..., perm[i], ...
        # New: ..., perm[i], perm[j], ..., perm[i-1], ...
        # i.e., perm[i] moves from position i to position j
    
    c_prev = np.array(machine_states[start], dtype=np.float64)
    
    for k in range(start, n_jobs):
        if i < j:
            # Position k in new sequence:
            # k from i to j-1: old position k+1
            # k == j: old position i (the moved job)
            # k > j: old position k
            if k == j:
                job = perm[i]
            elif k < j:
                job = perm[k + 1]
            else:
                job = perm[k]
        else:
            # Position k in new sequence:
            # k == j: old position i (the moved job)
            # k from j+1 to i: old position k-1
            # k > i: old position k
            if k == j:
                job = perm[i]
            elif k > j and k <= i:
                job = perm[k - 1]
            else:
                job = perm[k]
        
        p = processing_times[job]
        c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + p
        c_prev = c_curr
    
    return float(c_prev[-1])


def _neh_order(processing_times, seed=None):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling with optional randomization."""
    n_jobs, n_machines = processing_times.shape
    rng = random.Random(seed)
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # For larger instances, add some randomization to the order
    if seed is not None and n_jobs > 10:
        # Slightly shuffle the sorted order
        for _ in range(min(5, n_jobs // 10)):
            idx = rng.randint(0, n_jobs - 2)
            sorted_jobs[idx], sorted_jobs[idx+1] = sorted_jobs[idx+1], sorted_jobs[idx]
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _random_order(processing_times, seed=None):
    """Generate a random permutation."""
    n_jobs = processing_times.shape[0]
    perm = list(range(n_jobs))
    rng = random.Random(seed)
    rng.shuffle(perm)
    return perm


def _improve_order(perm, processing_times, time_limit=10.0, seed=None):
    """
    Improve permutation using insertion and swap moves with simulated annealing.
    Uses precomputed machine states for fast evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    rng = random.Random(seed)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    start_time = time.time()
    
    # Precompute machine states for all prefixes
    def compute_machine_states(p):
        states = []
        c_prev = np.zeros(processing_times.shape[1], dtype=np.float64)
        states.append(c_prev.copy())
        for j in range(len(p)):
            pt = processing_times[p[j]]
            c_curr = np.maximum(c_prev, np.concatenate(([0.0], c_prev[:-1]))) + pt
            c_prev = c_curr
            states.append(c_prev.copy())
        return states
    
    # Phase 1: Greedy local search with insertion moves (most powerful neighborhood)
    improved = True
    phase1_deadline = time_limit * 0.35
    while improved and (time.time() - start_time) < phase1_deadline:
        improved = False
        machine_states = compute_machine_states(current_perm)
        for i in range(n_jobs):
            if (time.time() - start_time) >= phase1_deadline:
                break
            for j in range(n_jobs):
                if i == j or abs(i - j) == 1:
                    continue
                ms = _evaluate_insert(current_perm, processing_times, i, j, machine_states)
                if ms < current_ms - 1e-9:
                    # Apply the insertion
                    job = current_perm[i]
                    del current_perm[i]
                    if i < j:
                        current_perm.insert(j - 1, job)
                    else:
                        current_perm.insert(j, job)
                    current_ms = ms
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best_perm = list(current_perm)
                    improved = True
                    break  # Restart scan from beginning after improvement
            if improved:
                break  # Restart outer loop
    
    # Phase 2: Simulated annealing with both swap and insertion moves
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        t0 = max(10.0, best_ms * 0.02)
        t_end = 0.1
        cooling_rate = 0.995
        
        temperature = t0
        no_improve_count = 0
        max_no_improve = 500
        
        while temperature > t_end and (time.time() - start_time) < time_limit:
            # Randomly choose between swap and insertion (60% insertion, 40% swap)
            use_insert = rng.random() < 0.6
            
            if use_insert:
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j or abs(i - j) == 1:
                    temperature *= cooling_rate
                    continue
            else:
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j:
                    temperature *= cooling_rate
                    continue
            
            machine_states = compute_machine_states(current_perm)
            
            if use_insert:
                candidate_ms = _evaluate_insert(current_perm, processing_times, i, j, machine_states)
                # Build candidate
                job = current_perm[i]
                del current_perm[i]
                if i < j:
                    candidate_perm = current_perm[:j-1] + [job] + current_perm[j-1:]
                else:
                    candidate_perm = current_perm[:j] + [job] + current_perm[j:]
                current_perm = candidate_perm
            else:
                candidate_ms = _evaluate_swap(current_perm, processing_times, i, j, machine_states)
                candidate_perm = list(current_perm)
                candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]
                current_perm = candidate_perm
            
            delta = candidate_ms - current_ms
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_ms = candidate_ms
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best_perm = list(current_perm)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                if use_insert:
                    # Revert insertion
                    job = current_perm[j if i > j else j - 1]
                    del current_perm[j if i > j else j - 1]
                    current_perm.insert(i, job)
                else:
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                no_improve_count += 1
            
            temperature *= cooling_rate
            
            # Early termination if stuck
            if no_improve_count > max_no_improve:
                break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH initialization, then insertion-based local search and SA for improvement.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (150s limit, use 135s)
    global_deadline = time.time() + 135.0
    
    # Generate multiple candidate permutations
    candidates = []
    
    # 1. Standard NEH
    neh_perm = _neh_order(processing_times, seed=None)
    candidates.append(neh_perm)
    
    # 2. NEH with slight randomization
    if n_jobs > 5:
        neh_rand_perm = _neh_order(processing_times, seed=42)
        if neh_rand_perm != neh_perm:
            candidates.append(neh_rand_perm)
    
    # 3. Random permutation
    rand_perm = _random_order(processing_times, seed=123)
    candidates.append(rand_perm)
    
    # 4. SPT (Shortest Processing Time) order
    totals = processing_times.sum(axis=1)
    spt_perm = list(np.argsort(totals))
    candidates.append(spt_perm)
    
    # Evaluate all candidates and pick the best
    best_perm = candidates[0]
    best_ms = _makespan(best_perm, processing_times)
    for cand in candidates[1:]:
        ms = _makespan(cand, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand
    
    # Improve with insertion-based local search + SA
    remaining = global_deadline - time.time()
    
    if remaining > 2.0:
        improve_time = min(remaining - 1.0, 120.0)
        improved_perm, improved_ms = _improve_order(
            best_perm, processing_times, time_limit=improve_time, seed=42
        )
        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms
    
    # If we still have time, try another round from a different start
    remaining = global_deadline - time.time()
    if remaining > 5.0 and n_jobs > 5:
        rand_perm2 = _random_order(processing_times, seed=999)
        improve_time2 = min(remaining - 1.0, 60.0)
        improved_perm2, improved_ms2 = _improve_order(
            rand_perm2, processing_times, time_limit=improve_time2, seed=777
        )
        if improved_ms2 < best_ms:
            best_perm = improved_perm2
            best_ms = improved_ms2
    
    return best_perm
# EVOLVE-BLOCK-END
