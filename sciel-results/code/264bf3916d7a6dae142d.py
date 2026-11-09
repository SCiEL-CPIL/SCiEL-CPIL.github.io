"""Improved program for flow-shop permutation construction using NEH + Fast Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Uses the standard dynamic programming recurrence:
    C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _completion_states(order, P):
    """
    Compute completion time states after each job.
    Returns list of states where states[k] = c_machines after processing first k+1 jobs.
    states[k][m] = completion time on machine m after job order[k].
    """
    n_machines = P.shape[1]
    n_jobs = len(order)
    states = []
    c_machines = [0.0] * n_machines
    
    for k in range(n_jobs):
        job = order[k]
        row = P[job]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
        states.append(c_machines[:])
        
    return states

def _makespan_from_states(states):
    """Get makespan from precomputed states."""
    if not states:
        return 0.0
    return states[-1][-1]

def _eval_insertion_fast(order, states, P, i, j):
    """
    Evaluate makespan if job at index i is moved to index j, using precomputed states.
    i and j are indices in the ORIGINAL order.
    Returns the new makespan.
    """
    n_machines = P.shape[1]
    n_jobs = len(order)
    
    if j < i:
        # Move job i to position j (before j)
        # Affected range: positions j through i in original
        # New segment: order[i], order[j], order[j+1], ..., order[i-1]
        # Start state: states[j-1] if j > 0, else zeros
        if j == 0:
            c_machines = [0.0] * n_machines
        else:
            c_machines = states[j-1][:]
        
        # Process the new segment
        # Jobs: order[i], then order[j] through order[i-1]
        job_moved = order[i]
        row = P[job_moved]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
        
        for k in range(j, i):
            job = order[k]
            row = P[job]
            c_curr = 0.0
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
        
        # Continue with remaining jobs from i+1 to n-1
        for k in range(i+1, n_jobs):
            job = order[k]
            row = P[job]
            c_curr = 0.0
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
        
        return c_machines[-1]
    
    else:
        # Move job i to position j (after j-1, so at position j in new sequence)
        # When i < j: moving job i to position j means it goes after position j-1 in new seq
        # Original: ..., order[i], ..., order[j], ...
        # New: ..., order[i+1], ..., order[j], order[i], ...
        # Wait, let me think again.
        # If we remove job at index i and insert at index j (in the new sequence),
        # and j > i:
        # Original: A B C D E, i=1 (B), j=3
        # Remove B: A C D E
        # Insert B at index 3: A C D B E
        # So B ends up at position 3 in new sequence.
        # Affected range: positions i through j in original
        # New segment: order[i+1], order[i+2], ..., order[j], order[i]
        # Start state: states[i-1] if i > 0, else zeros
        
        if i == 0:
            c_machines = [0.0] * n_machines
        else:
            c_machines = states[i-1][:]
        
        # Process the new segment
        for k in range(i+1, j+1):
            job = order[k]
            row = P[job]
            c_curr = 0.0
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
        
        # Now process the moved job
        job_moved = order[i]
        row = P[job_moved]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
        
        # Continue with remaining jobs from j+1 to n-1
        for k in range(j+1, n_jobs):
            job = order[k]
            row = P[job]
            c_curr = 0.0
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
        
        return c_machines[-1]

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _random_order(n_jobs):
    """Generate a random permutation."""
    order = list(range(n_jobs))
    random.shuffle(order)
    return order

def _local_search(order, P, deadline):
    """
    First-improvement local search with insertion moves.
    Uses precomputed completion states for faster evaluation.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_jobs <= 1:
        return order, _makespan(order, P)
    
    best_order = order[:]
    best_makespan = _makespan(best_order, P)
    states = _completion_states(best_order, P)
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        
        for i in range(n_jobs):
            if (time.time() > deadline):
                break
            job_i = best_order[i]
            
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Check time more frequently
                if (time.time() > deadline):
                    break
                
                ms = _eval_insertion_fast(best_order, states, P, i, j)
                
                if ms < best_makespan:
                    best_makespan = ms
                    best_order = best_order[:]
                    best_order.pop(i)
                    best_order.insert(j, job_i)
                    states = _completion_states(best_order, P)
                    improved = True
                    break
                    
            if improved:
                break
                
    return best_order, best_makespan

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH heuristic followed by local search.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # Contract says 120s limit. Use 115s safety margin.
    deadline = start_time + 115.0
    
    # Generate diverse initial solutions
    initial_orders = []
    
    # NEH
    neh_order = _neh_heuristic(P)
    initial_orders.append(neh_order)
    
    # SPT (Shortest Processing Time) - sort by total processing time ascending
    totals = P.sum(axis=1)
    spt_order = list(np.argsort(totals))
    initial_orders.append(spt_order)
    
    # LPT is same as NEH initial sort, skip
    
    # Random orders
    n_random = 5
    for _ in range(n_random):
        initial_orders.append(_random_order(n_jobs))
    
    # Evaluate all initial solutions and pick the best
    best_order = None
    best_makespan = float('inf')
    
    for order in initial_orders:
        ms = _makespan(order, P)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order[:]
    
    # Local search on the best initial solution
    best_order, best_makespan = _local_search(best_order, P, deadline)
    
    # If we have time, try local search on other initial solutions
    for order in initial_orders:
        if (time.time() > deadline):
            break
        if order == best_order:
            continue
        ms = _makespan(order, P)
        if ms < best_makespan:
            # This initial solution is better, local search it
            new_order, new_ms = _local_search(order, P, deadline)
            if new_ms < best_makespan:
                best_makespan = new_ms
                best_order = new_order
        elif (time.time() > deadline):
            break
    
    return best_order
# EVOLVE-BLOCK-END
