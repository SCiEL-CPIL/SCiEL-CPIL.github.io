"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps


def _compute_suffix_completions(perm, processing_times):
    """
    Compute suffix delay vectors using max-plus algebra.
    suffix_comps[i] stores vector D such that Makespan(V, perm[i:]) = max_k(V[k] + D[k]).
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)]
    
    suffix_comps = [None] * (n_jobs + 1)
    suffix_comps[n_jobs] = np.zeros(n_machines)
    
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        D_next = suffix_comps[i+1]
        
        # Compute W: coefficients for completion vector after processing job `job`
        # W[m] = coefficient of V[m] in C_m (completion on machine m)
        W = np.full(n_machines, -np.inf)
        W[0] = pt[0]
        
        for m in range(1, n_machines):
            W += pt[m]
            # Ensure W[m] is at least pt[m] (handles -inf case)
            if W[m] < pt[m]:
                W[m] = pt[m]
        
        # D_current[k] = max_j (W[k] + D_next[j]) = W[k] + max_j(D_next[j])
        max_D_next = np.max(D_next)
        suffix_comps[i] = W + max_D_next
        
    return suffix_comps


def _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1].
    i is the index of the first job in the pair.
    """
    n_machines = processing_times.shape[1]
    
    # Get completion vector before perm[i]
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]
    
    job_a = perm[i]
    job_b = perm[i+1]
    
    # Simulate job_b then job_a
    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]
    
    # C_b
    C_b = np.zeros(n_machines)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = max(C_prev[m], C_b[m-1]) + pt_b[m]
        
    # C_a
    C_a = np.zeros(n_machines)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = max(C_b[m], C_a[m-1]) + pt_a[m]
        
    # Now evaluate suffix starting from i+2
    # The suffix is perm[i+2..]
    # We need to combine C_a with suffix_comps[i+2]
    
    if i + 2 >= len(perm):
        return float(np.max(C_a))
    else:
        D_suffix = suffix_comps[i+2]
        # Makespan = max_k (C_a[k] + D_suffix[k])
        return float(np.max(C_a + D_suffix))


def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    If i < j, the job is moved to j (0-indexed in the new list, which corresponds to 
    inserting after perm[j-1] in the old list? No, standard insertion).
    
    Let's define the move clearly.
    Original: ... A ... B ...
    Remove A at i.
    Insert A at j.
    
    If i < j:
    Sequence: P[0..i-1] + P[i+1..j] + A + P[j+1..]
    If i > j:
    Sequence: P[0..j-1] + A + P[j..i-1] + P[i+1..]
    
    Note: The "position j" usually means the index in the resulting list.
    If we remove i, the list shrinks.
    If we insert at j, it's the index in the list *after* removal?
    Standard convention in these loops:
    `new_perm = _move_job(perm, job, new_pos)`
    `_move_job` removes job, then inserts at `new_pos` in the reduced list.
    
    So if i < j:
    The job moves to the right.
    Prefix: P[0..i-1]. Completion C_{i-1}.
    Middle: P[i+1..j]. (Jobs from i+1 to j in original indices).
    Job A.
    Suffix: P[j+1..].
    
    Wait, if we remove i, the element at index j in the *original* list shifts to j-1.
    The insertion position `j` in the *new* list corresponds to:
    - If j > i: The job is inserted after the element that was at index j-1 in the original list?
      Let's trace `_move_job`:
      `perm = [j for j in perm if j != job]`
      `return perm[:new_pos] + [job] + perm[new_pos:]`
      
      If i=1, j=3.
      Original: [0, 1, 2, 3]
      Remove 1: [0, 2, 3]
      Insert at 3: [0, 2, 3, 1]
      So the sequence is 0, 2, 3, 1.
      Original indices: 0, 2, 3, 1.
      Prefix: 0. (i-1 = 0). C_0.
      Middle: 2, 3. (Original indices i+1 to j).
      Job: 1.
      Suffix: None.
      
      If i=3, j=1.
      Original: [0, 1, 2, 3]
      Remove 3: [0, 1, 2]
      Insert at 1: [0, 3, 1, 2]
      Sequence: 0, 3, 1, 2.
      Original indices: 0, 1, 2, 3.
      Prefix: 0. (j-1 = 0). C_0.
      Job: 3.
      Middle: 1, 2. (Original indices j to i-1).
      Suffix: None.
    """
    n_machines = processing_times.shape[1]
    
    if i == j:
        return _makespan(perm, processing_times) # Should not happen
    
    if i < j:
        # Move right
        # Prefix: 0..i-1
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1]
            
        # Middle: i+1..j
        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        # Job A (perm[i])
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_mid[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_mid[m], C_a[m-1]) + pt_a[m]
            
        # Suffix: j+1..
        if j + 1 >= len(perm):
            return float(np.max(C_a))
        else:
            D_suffix = suffix_comps[j+1]
            return float(np.max(C_a + D_suffix))
            
    else:
        # Move left (i > j)
        # Prefix: 0..j-1
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1]
            
        # Job A (perm[i])
        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_prev[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_prev[m], C_a[m-1]) + pt_a[m]
            
        # Middle: j..i-1
        C_mid = C_a
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new
            
        # Suffix: i+1..
        if i + 1 >= len(perm):
            return float(np.max(C_mid))
        else:
            D_suffix = suffix_comps[i+1]
            return float(np.max(C_mid + D_suffix))


def _makespan_from_prev(perm, processing_times, prev_completion):
    """
    Compute makespan starting from a known completion vector for a prefix.
    prev_completion[m] is the completion time on machine m after some prefix.
    perm is the full permutation, and we compute from scratch but this is
    just the same as _makespan. Keeping for clarity.
    """
    return _makespan(perm, processing_times)


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        # Swap a few adjacent elements randomly
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    # Backward pass: remove each job and reinsert at best position
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        # Remove job at position i
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _insert_job(perm, job, pos):
    """Insert job at position pos in perm (removing it from current position if present)."""
    if job in perm:
        perm = [j for j in perm if j != job]
    return perm[:pos] + [job] + perm[pos:]


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _local_search_2opt(perm, processing_times, max_iterations=200, deadline=None):
    """
    Local search with best-improvement strategy using incremental evaluation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Precompute prefix and suffix vectors for current best_perm
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)
        
        best_improvement_ms = best_ms
        best_move = None
        
        # Try adjacent swaps - find best improvement
        for i in range(n_jobs - 1):
            ms = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            if ms < best_improvement_ms - 1e-10:
                best_improvement_ms = ms
                best_move = ('swap', i, None)
        
        # Try insert moves - find best improvement
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                if ms < best_improvement_ms - 1e-10:
                    best_improvement_ms = ms
                    best_move = ('insert', i, pos)
        
        # Apply best move if found
        if best_move is not None:
            move_type, i, pos = best_move
            if move_type == 'swap':
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
            else:
                job = best_perm[i]
                best_perm = _move_job(best_perm, job, pos)
            best_ms = best_improvement_ms
        else:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with incremental evaluation for swap moves.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    initial_temp = current_ms * 0.1
    cooling_rate = 0.9995
    
    temp = initial_temp
    step = 0
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        step += 1
        
        # Precompute prefix/suffix every 50 steps or at start
        if step == 1 or step % 50 == 0:
            prefix_comps = _compute_prefix_completions(current_perm, processing_times)
            suffix_comps = _compute_suffix_completions(current_perm, processing_times)
        
        # Generate neighbor - favor adjacent swaps for incremental eval
        move_type = rng.random()
        if move_type < 0.7:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
            new_ms = _eval_swap_incremental(current_perm, i, processing_times, prefix_comps, suffix_comps)
        else:
            # Insert move - use full makespan (less frequent)
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
            new_ms = _makespan(new_perm, processing_times)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. Simple LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    # 3. NEH with randomization (diversified)
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    # 4. NEH forward-backward
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Local search polish on best solution - use more time
    if time.time() < deadline and n_jobs > 1:
        remaining = deadline - time.time()
        # Allocate more time to local search
        ls_deadline = deadline - max(5.0, remaining * 0.2)
        
        if n_jobs <= 30:
            max_iter = 2000
        elif n_jobs <= 50:
            max_iter = 1000
        elif n_jobs <= 100:
            max_iter = 500
        else:
            max_iter = 200
        
        polished = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter, deadline=ls_deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing if time remains
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search polish
        if time.time() < deadline - 2.0:
            max_iter_final = 500 if n_jobs <= 50 else 200
            final_polish = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 2.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
