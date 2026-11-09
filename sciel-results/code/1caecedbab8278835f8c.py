"""Combined program: NEH with Iterated Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(order, P):
    """
    Compute makespan using a fast list-based approach.
    """
    if not order:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0

    # Use Python lists for the inner loop
    job0 = order[0]
    curr_row = [0.0] * n_machines
    curr_row[0] = P[job0, 0]
    for m in range(1, n_machines):
        curr_row[m] = curr_row[m-1] + P[job0, m]
        
    for idx in range(1, n_jobs):
        job = order[idx]
        prev_row = curr_row
        new_row = [0.0] * n_machines
        new_row[0] = prev_row[0] + P[job, 0]
        for m in range(1, n_machines):
            val_prev = prev_row[m]
            val_left = new_row[m-1]
            new_row[m] = (val_prev if val_prev > val_left else val_left) + P[job, m]
        curr_row = new_row
        
    return curr_row[-1]

def _neh_heuristic(P, deadline=None):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    totals = P.sum(axis=1)
    # Add small random noise to break ties
    noise = np.random.uniform(0, 0.1 * np.max(totals), size=n_jobs) if n_jobs > 1 else np.array([0.0])
    sorted_indices = np.argsort(-(totals + noise))
    
    current_seq = [int(sorted_indices[0])]
    
    for k in range(1, n_jobs):
        if deadline is not None and time.time() > deadline:
            break
            
        job = int(sorted_indices[k])
        best_pos = -1
        best_ms = float('inf')
        
        for pos in range(len(current_seq) + 1):
            if deadline is not None and pos % 5 == 0 and time.time() > deadline:
                break
                
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan_fast(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        if best_pos == -1:
            current_seq.append(job)
        else:
            current_seq.insert(best_pos, job)
            
    return current_seq

def _sa_run(order, P, deadline):
    """
    Run Simulated Annealing until deadline.
    Returns (best_order, best_ms).
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan_fast(order, P)
    
    current_order = list(order)
    current_ms = _makespan_fast(current_order, P)
    best_order = list(current_order)
    best_ms = current_ms
    
    # SA Parameters
    # Initial temperature
    T = max(1.0, current_ms * 0.1)
    T_min = 1e-6
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Choose a move
        if random.random() < 0.5:
            # Swap move
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insert move
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
            job = current_order.pop(i)
            if i < j:
                j -= 1
            if j < 0: j = 0
            if j > n_jobs - 1: j = n_jobs - 1
            new_order = current_order.copy()
            new_order.insert(j, job)
            
        new_ms = _makespan_fast(new_order, P)
        
        # Acceptance probability
        delta = new_ms - current_ms
        if delta < 0:
            current_order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        else:
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_order = new_order
                    current_ms = new_ms
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_order, best_ms

def _iterated_sa(order, P, deadline):
    """
    Iterated Simulated Annealing.
    """
    n_jobs = len(order)
    best_order = list(order)
    best_ms = _makespan_fast(best_order, P)
    current_order = list(best_order)
    
    while time.time() < deadline:
        # Run SA for a short burst (1 second)
        burst_deadline = min(time.time() + 1.0, deadline)
        refined_order, refined_ms = _sa_run(current_order, P, burst_deadline)
        
        if refined_ms < best_ms:
            best_ms = refined_ms
            best_order = refined_order
            current_order = refined_order
        else:
            # Perturb best_order
            current_order = list(best_order)
            # Swap 2-5 random pairs
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i != j:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    
    return best_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    # Safety margin: 140 seconds
    deadline = start_time + 140.0
    
    # 1. NEH Heuristic
    permutation = _neh_heuristic(P, deadline)
    
    # Ensure permutation is complete
    if len(permutation) < n_jobs:
        remaining = [j for j in range(n_jobs) if j not in permutation]
        totals = P.sum(axis=1)
        remaining.sort(key=lambda x: -totals[x])
        permutation.extend(remaining)
        
    # 2. Iterated Simulated Annealing
    sa_deadline = deadline - 5.0
    if time.time() < sa_deadline:
        permutation = _iterated_sa(permutation, P, deadline=sa_deadline)
        
    return list(permutation)

# EVOLVE-BLOCK-END
