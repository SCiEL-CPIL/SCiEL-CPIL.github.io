"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import copy


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation of job indices."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = completion[m] + processing_times[job, m]
    return completion[-1]


def _neh_heuristic(processing_times, order):
    """NEH constructive heuristic given an initial job order."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    # Start with empty schedule, insert jobs one by one in given order
    schedule = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(schedule) + 1):
            # Try inserting job at position pos
            candidate = schedule[:pos] + [job] + schedule[pos:]
            # Compute makespan efficiently
            completion = np.zeros(n_machines)
            for j in candidate:
                for m in range(n_machines):
                    completion[m] += processing_times[j, m]
            mk = completion[-1]
            if mk < best_makespan:
                best_makespan = mk
                best_pos = pos
        schedule = schedule[:best_pos] + [job] + schedule[best_pos:]
    return schedule


def _local_search_2opt(perm, processing_times, max_iters=100, deadline=None):
    """2-opt local search: try all adjacent swaps and reverse segments."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _makespan(best_perm, processing_times)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            mk = _makespan(new_perm, processing_times)
            if mk < best_makespan - 1e-9:
                best_perm = new_perm
                best_makespan = mk
                improved = True
        # Try 2-opt reversals (reverse a segment)
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 2, n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                new_perm = list(best_perm[:i]) + list(reversed(best_perm[i:j+1])) + list(best_perm[j+1:])
                mk = _makespan(new_perm, processing_times)
                if mk < best_makespan - 1e-9:
                    best_perm = new_perm
                    best_makespan = mk
                    improved = True
    return best_perm, best_makespan


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using a portfolio search."""
    start_time = time.time()
    # Safety margin: stop searching at 120s, return before 130s
    search_deadline = start_time + 120.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate diverse initial orderings
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    spt_order = list(np.argsort(processing_times.sum(axis=1)))
    random.seed(42)
    random_orders = []
    for _ in range(3):
        o = list(range(n_jobs))
        random.shuffle(o)
        random_orders.append(o)
    
    # Candidate initial solutions
    initial_solutions = [lpt_order, spt_order] + random_orders
    
    best_perm = None
    best_makespan = float('inf')
    
    for init_order in initial_solutions:
        if time.time() > search_deadline:
            break
        
        # Apply NEH heuristic
        neh_schedule = _neh_heuristic(processing_times, init_order)
        neh_makespan = _makespan(neh_schedule, processing_times)
        
        if neh_makespan < best_makespan:
            best_makespan = neh_makespan
            best_perm = list(neh_schedule)
        
        # Apply local search
        ls_perm, ls_makespan = _local_search_2opt(
            neh_schedule, processing_times,
            max_iters=50, deadline=search_deadline
        )
        
        if ls_makespan < best_makespan:
            best_makespan = ls_makespan
            best_perm = list(ls_perm)
    
    # If time remains, do additional local search passes on the best solution
    if best_perm is not None and time.time() < search_deadline:
        for _ in range(5):
            if time.time() > search_deadline:
                break
            ls_perm, ls_makespan = _local_search_2opt(
                best_perm, processing_times,
                max_iters=20, deadline=search_deadline
            )
            if ls_makespan < best_makespan:
                best_makespan = ls_makespan
                best_perm = list(ls_perm)
            else:
                break
    
    return best_perm if best_perm is not None else list(range(n_jobs))
# EVOLVE-BLOCK-END
