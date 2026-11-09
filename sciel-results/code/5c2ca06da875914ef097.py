"""Improved program for flow-shop permutation construction using NEH + Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    # completion[m] = completion time of last scheduled job on machine m
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _neh_heuristic(P):
    """NEH heuristic: build permutation by inserting each job at best position."""
    n_jobs = P.shape[0]
    totals = P.sum(axis=1)
    # Sort jobs by descending total processing time
    order = list(np.argsort(-totals))
    
    # Greedily insert each job at the position minimizing makespan
    perm = []
    for job in order:
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            mk = _makespan(test_perm, P)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _two_opt(perm, P, max_iterations=1000, deadline=None):
    """2-Opt local search: swap pairs of jobs if it improves makespan."""
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = list(perm)
    best_mk = _makespan(best_perm, P)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Reverse segment between i and j (2-opt move)
                new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                mk = _makespan(new_perm, P)
                if mk < best_mk - 1e-10:
                    best_perm = new_perm
                    best_mk = mk
                    improved = True
                    # Reset j to continue exploring from this new position
                    j = i + 1
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using NEH + 2-Opt."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs = P.shape[0]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH heuristic to get a good initial solution
    perm = _neh_heuristic(P)
    
    # Step 2: 2-Opt local search to refine
    perm = _two_opt(perm, P, max_iterations=500, deadline=deadline)
    
    # If we still have time, do additional 2-Opt passes
    if time.time() < deadline - 5.0:
        perm = _two_opt(perm, P, max_iterations=300, deadline=deadline)
    
    return list(perm)
# EVOLVE-BLOCK-END
