"""Improved program for flow-shop permutation construction using NEH + Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    
    C = np.zeros(n_machines)
    for j in order:
        prev = 0
        for m in range(n_machines):
            val = max(C[m], prev) + P[j, m]
            C[m] = val
            prev = val
    
    return C[-1]


def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow shop.
    Sort jobs by total processing time (descending), then insert each job
    at the best position in the current partial sequence.
    """
    n_jobs, n_machines = P.shape
    
    # Sort by total processing time, descending
    totals = P.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence greedily
    seq = [order[0]]
    for idx in range(1, n_jobs):
        job = order[idx]
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            # Insert job at position pos
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _local_search(order, P, deadline):
    """
    Local search using adjacent swaps (best improvement) for large n,
    and insertion moves for small n.
    """
    n = len(order)
    if n <= 1:
        return list(order), _makespan(order, P)
        
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # Determine strategy based on problem size
    # Insertion is O(n^3 * m) roughly, Adjacent is O(n^2 * m) per pass.
    # For n > 20, adjacent swaps with best improvement is safer for time.
    
    if n <= 20:
        # Full insertion local search
        improved = True
        while improved:
            improved = False
            best_move = None
            best_move_ms = current_ms
            
            for i in range(n):
                if time.time() > deadline - 5:
                    return current_order, current_ms
                
                job = current_order[i]
                reduced = current_order[:i] + current_order[i+1:]
                
                for k in range(len(reduced) + 1):
                    new_order = reduced[:k] + [job] + reduced[k:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < best_move_ms:
                        best_move_ms = new_ms
                        best_move = new_order
            
            if best_move is not None:
                current_order = best_move
                current_ms = best_move_ms
                improved = True
    else:
        # Adjacent swap local search with best improvement
        improved = True
        while improved:
            improved = False
            best_swap_i = -1
            best_swap_ms = current_ms
            
            for i in range(n - 1):
                if time.time() > deadline - 5:
                    return current_order, current_ms
                
                new_order = current_order.copy()
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                new_ms = _makespan(new_order, P)
                
                if new_ms < best_swap_ms:
                    best_swap_ms = new_ms
                    best_swap_i = i
            
            if best_swap_i != -1:
                current_order[best_swap_i], current_order[best_swap_i+1] = \
                    current_order[best_swap_i+1], current_order[best_swap_i]
                current_ms = best_swap_ms
                improved = True

    return current_order, current_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic followed by local search with a time deadline.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: NEH heuristic
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # Phase 2: Local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:  # leave 5s safety margin
        current_order, current_ms = _local_search(best_order, P, deadline)
        if current_ms < best_ms:
            best_order = current_order
            best_ms = current_ms
    
    # Phase 3: Additional restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random restarts with perturbation
        rng = np.random.default_rng(42)
        for restart in range(5):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # Perturb the best solution: random swap of two positions
            perturbed = list(best_order)
            i, j = rng.choice(n_jobs, size=2, replace=False)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            # Local search from perturbed solution
            local_order, local_ms = _local_search(perturbed, P, deadline)
            if local_ms < best_ms:
                best_order = local_order
                best_ms = local_ms
    
    return list(best_order)
# EVOLVE-BLOCK-END
