# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order using the flow-shop recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    # c[j] = completion time on machine j for the last job processed
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] += p_row[j]
            else:
                c[j] = max(c[j], c[j - 1]) + p_row[j]
    return c[-1]


def _makespan_from_array(pt, order_arr):
    """Vectorized-friendly makespan computation where order_arr is a numpy array of job indices."""
    n_machines = pt.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order_arr:
        p_row = pt[int(job_idx)]
        for j in range(n_machines):
            if j == 0:
                c[j] += p_row[j]
            else:
                c[j] = max(c[j], c[j - 1]) + p_row[j]
    return c[-1]


def _johnson_two_machine(pt):
    """
    Apply Johnson's rule to aggregate first and last machines.
    Returns a permutation of job indices.
    """
    n_jobs = pt.shape[0]
    if n_jobs == 0:
        return []
    
    # Aggregate: for each job, sum of processing on first machine and last machine
    a = pt[:, 0].copy()  # first machine
    b = pt[:, -1].copy()  # last machine
    
    # Johnson's rule: partition into two sets
    # Set 1: jobs where a_i < b_i, sorted by a_i ascending
    # Set 2: jobs where a_i >= b_i, sorted by b_i descending
    # Permutation = Set1 + Set2
    
    set1 = [i for i in range(n_jobs) if a[i] < b[i]]
    set2 = [i for i in range(n_jobs) if a[i] >= b[i]]
    
    set1.sort(key=lambda i: a[i])
    set2.sort(key=lambda i: -b[i])
    
    return set1 + set2


def _neh_rule(pt):
    """
    NEH (Nawaz-Enscore-Ham) heuristic: build sequence by inserting each job
    (sorted by total processing time descending) at the best position.
    """
    n_jobs = pt.shape[0]
    if n_jobs == 0:
        return []
    
    totals = pt.sum(axis=1)
    order_sorted = list(np.argsort(-totals))
    
    seq = [order_sorted[0]]
    for job in order_sorted[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(pt, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    
    return seq


def _insertion_local_search(pt, order, max_passes=10, deadline=None):
    """
    First-improvement insertion local search.
    For each job, try inserting it at every other position; take the best improving move.
    Use vectorized evaluation where possible.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(pt, current_order)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            
            job = current_order[i]
            # Remove job at position i
            remaining = current_order[:i] + current_order[i+1:]
            
            best_pos = i
            best_ms = current_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(pt, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < current_ms:
                # Apply the best move
                new_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_order = new_order
                current_ms = best_ms
                improved = True
        
        if not improved:
            break
    
    return current_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Generate initial permutations using Johnson's rule (on first/last machines) and NEH.
    2. Evaluate both, keep the better.
    3. Apply insertion-based local search with a time deadline.
    4. Return the best permutation found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs = pt.shape[0]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate candidate initial permutations
    candidates = []
    
    # Johnson's rule on first and last machines
    johnson_perm = _johnson_two_machine(pt)
    candidates.append(johnson_perm)
    
    # NEH rule
    neh_perm = _neh_rule(pt)
    candidates.append(neh_perm)
    
    # Also try Johnson's rule on reversed machines (last as "first", first as "last")
    pt_rev = pt[:, ::-1]
    johnson_rev_perm = _johnson_two_machine(pt_rev)
    candidates.append(johnson_rev_perm)
    
    # Longest total processing time first (baseline)
    lpt_perm = list(np.argsort(-pt.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # Shortest total processing time first
    spt_perm = list(np.argsort(pt.sum(axis=1)))
    candidates.append(spt_perm)
    
    # Evaluate all candidates and pick the best
    best_order = None
    best_ms = float('inf')
    
    for perm in candidates:
        if deadline is not None and time.time() > deadline:
            break
        ms = _makespan(pt, perm)
        if ms < best_ms:
            best_ms = ms
            best_order = list(perm)
    
    if best_order is None:
        best_order = list(range(n_jobs))
        best_ms = _makespan(pt, best_order)
    
    # Apply insertion local search
    if time.time() < deadline:
        refined = _insertion_local_search(pt, best_order, max_passes=50, deadline=deadline)
        refined_ms = _makespan(pt, refined)
        if refined_ms < best_ms:
            best_order = refined
            best_ms = refined_ms
    
    # If we still have time, try local search from other good starting points
    if time.time() < deadline - 5.0:
        for perm in candidates:
            if time.time() > deadline - 5.0:
                break
            if list(perm) == best_order:
                continue
            perm_ms = _makespan(pt, perm)
            # Only refine if this starting point is reasonably good
            if perm_ms < best_ms * 1.1:
                refined2 = _insertion_local_search(pt, perm, max_passes=20, deadline=deadline)
                refined2_ms = _makespan(pt, refined2)
                if refined2_ms < best_ms:
                    best_order = refined2
                    best_ms = refined2_ms
    
    return list(best_order)
# EVOLVE-BLOCK-END
