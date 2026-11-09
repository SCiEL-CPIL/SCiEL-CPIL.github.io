"""Improved program for flow-shop permutation construction using NEH variants + Or-opt + SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


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


def _makespan_fast(order, P):
    """Faster makespan using precomputed job arrays."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    
    C = np.zeros(n_machines)
    for j in order:
        p = P[j]
        prev = 0.0
        for m in range(n_machines):
            cm = C[m]
            if prev > cm:
                val = prev + p[m]
            else:
                val = cm + p[m]
            C[m] = val
            prev = val
    
    return C[-1]


def _neh_heuristic(P, variant=0):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow shop.
    variant 0: total processing time descending
    variant 1: first machine processing time descending  
    variant 2: last machine processing time descending
    variant 3: random order
    """
    n_jobs, n_machines = P.shape
    
    if variant == 0:
        totals = P.sum(axis=1)
        order = list(np.argsort(-totals))
    elif variant == 1:
        first = P[:, 0]
        order = list(np.argsort(-first))
    elif variant == 2:
        last = P[:, -1]
        order = list(np.argsort(-last))
    else:
        rng = np.random.default_rng(variant + 100)
        order = list(rng.permutation(n_jobs))
    
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


def _insertion_search(order, P, deadline):
    """
    Local search using insertion moves (best improvement).
    """
    n = len(order)
    if n <= 1:
        return list(order), _makespan(order, P)
        
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
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

    return current_order, current_ms


def _or_opt_search(order, P, deadline):
    """
    Local search using Or-opt block moves (move a block of 1-3 jobs).
    """
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
        
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    while improved:
        improved = False
        best_move = None
        best_move_ms = current_ms
        
        for block_len in [1, 2, 3]:
            if time.time() > deadline - 5:
                return current_order, current_ms
            
            for i in range(n - block_len + 1):
                block = current_order[i:i+block_len]
                reduced = current_order[:i] + current_order[i+block_len:]
                
                for k in range(len(reduced) + 1):
                    if k == i:
                        continue
                    new_order = reduced[:k] + block + reduced[k:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < best_move_ms:
                        best_move_ms = new_ms
                        best_move = new_order
        
        if best_move is not None:
            current_order = best_move
            current_ms = best_move_ms
            improved = True

    return current_order, current_ms


def _simulated_annealing(order, P, deadline, rng):
    """
    Simulated annealing with swap and insertion moves.
    """
    n = len(order)
    if n <= 1:
        return list(order), _makespan(order, P)
        
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    best_order = list(current_order)
    best_ms = current_ms
    
    t0 = time.time()
    initial_temp = current_ms * 0.1 + 1.0
    cooling_rate = 0.95
    
    while time.time() < deadline - 5:
        temp = initial_temp * (cooling_rate ** int((time.time() - t0) * 10))
        if temp < 1e-6:
            temp = 1e-6
        
        # Choose move type
        if rng.random() < 0.5:
            # Swap
            i, j = rng.choice(n, size=2, replace=False)
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insertion
            i = rng.integers(n)
            j = rng.integers(n)
            new_order = current_order.copy()
            job = new_order.pop(i)
            new_order.insert(j, job)
        
        new_ms = _makespan(new_order, P)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multiple NEH variants, insertion/Or-opt local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = 140.0  # seconds, with safety margin before 150s kill
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try multiple NEH variants
    best_order = None
    best_ms = np.inf
    
    for variant in range(4):
        if time.time() > deadline - 20:
            break
        order = _neh_heuristic(P, variant=variant)
        ms = _makespan(order, P)
        if ms < best_ms:
            best_ms = ms
            best_order = order
    
    if best_order is None:
        best_order = list(range(n_jobs))
        best_ms = _makespan(best_order, P)
    
    # Phase 2: Insertion local search on best NEH solution
    if time.time() < deadline - 15:
        current_order, current_ms = _insertion_search(best_order, P, deadline)
        if current_ms < best_ms:
            best_order = current_order
            best_ms = current_ms
    
    # Phase 3: Or-opt local search
    if time.time() < deadline - 10:
        current_order, current_ms = _or_opt_search(best_order, P, deadline)
        if current_ms < best_ms:
            best_order = current_order
            best_ms = current_ms
    
    # Phase 4: Simulated annealing from best solution
    if time.time() < deadline - 5:
        sa_order, sa_ms = _simulated_annealing(best_order, P, deadline, rng)
        if sa_ms < best_ms:
            best_order = sa_order
            best_ms = sa_ms
    
    # Phase 5: More NEH variants + local search if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for variant in range(4, 8):
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
                break
            
            order = _neh_heuristic(P, variant=variant)
            ms = _makespan(order, P)
            
            # Quick local search
            if time.time() < deadline - 10:
                local_order, local_ms = _insertion_search(order, P, deadline)
                if local_ms < ms:
                    ms = local_ms
                    order = local_order
            
            if ms < best_ms:
                best_ms = ms
                best_order = order
    
    # Final polish with Or-opt if time permits
    if time.time() < deadline - 5:
        final_order, final_ms = _or_opt_search(best_order, P, deadline)
        if final_ms < best_ms:
            best_order = final_order
            best_ms = final_ms
    
    return list(best_order)
# EVOLVE-BLOCK-END
