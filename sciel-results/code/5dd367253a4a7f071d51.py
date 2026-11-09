"""Improved flow-shop permutation construction using NEH + 2-opt/Or-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_from_positions(positions, P):
    """Compute makespan given positions array where positions[j] = position of job j."""
    order = np.argsort(positions)
    return _makespan(order, P)


def _neh_heuristic(P):
    """NEH heuristic: build permutation by inserting each job at best position."""
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    totals = P.sum(axis=1)
    # Sort jobs by descending total processing time
    sorted_jobs = np.argsort(-totals)
    
    # Build permutation incrementally
    perm = []
    for j in sorted_jobs:
        # Try inserting job j at each possible position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, j)
    
    return perm


def _two_opt(order, P, max_iter=1000):
    """2-opt local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return order, _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment i..j
                candidate = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_order, best_ms


def _or_opt(order, P, max_iter=500):
    """Or-opt local search: try moving 1 or 2 consecutive jobs to another position."""
    n = len(order)
    if n <= 3:
        return order, _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for length in [1, 2]:
            for i in range(n - length + 1):
                segment = best_order[i:i+length]
                # Remove segment
                remaining = best_order[:i] + best_order[i+length:]
                # Try inserting at each position in remaining
                for j in range(len(remaining) - length + 1):
                    if j == i:
                        continue
                    candidate = remaining[:j] + segment + remaining[j:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 150s limit
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    # Start with NEH heuristic
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # Apply 2-opt local search
    best_order, best_ms = _two_opt(best_order, P, max_iter=200)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Apply Or-opt local search
    best_order, best_ms = _or_opt(best_order, P, max_iter=100)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Second pass of 2-opt if time allows
    if time.time() < deadline - 20:
        best_order, best_ms = _two_opt(best_order, P, max_iter=100)
    
    # Final Or-opt pass if time allows
    if time.time() < deadline - 10:
        best_order, best_ms = _or_opt(best_order, P, max_iter=50)
    
    return best_order
# EVOLVE-BLOCK-END
