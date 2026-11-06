"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, p):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0
    # Cumulative completion times
    # completion[m] = completion time of last job on machine m
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            if m == 0:
                completion[m] += p[job, m]
            else:
                completion[m] = max(completion[m - 1], completion[m]) + p[job, m]
    return completion[-1]


def _makespan_fast(order, p):
    """Faster makespan computation using numpy."""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0
    # Build submatrix for this order
    sub = p[order, :]
    # DP: C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # C[j, 0] = sum_{i<=j} p[i, 0]
    C = np.zeros((n_jobs, n_machines))
    C[:, 0] = np.cumsum(sub[:, 0])
    for m in range(1, n_machines):
        for j in range(n_jobs):
            C[j, m] = max(C[j - 1, m] if j > 0 else 0.0, C[j, m - 1]) + sub[j, m]
    return C[-1, -1]


def _neh_initialization(p):
    """NEH heuristic: insert jobs one by one in position minimizing makespan."""
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return []
    
    totals = p.sum(axis=1)
    order_by_total = np.argsort(-totals)
    
    current_order = [int(order_by_total[0])]
    for k in range(1, n_jobs):
        job = int(order_by_total[k])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(candidate, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _adjacent_swap_delta(order, i, p):
    """Compute makespan change when swapping adjacent jobs at positions i and i+1."""
    n_jobs, n_machines = p.shape
    # Compute completion times up to position i-1
    # This is expensive; instead compute full makespan before and after
    # For efficiency in local search, we can use incremental computation
    
    # Simple approach: compute makespan for both orders
    ms_before = _makespan_fast(order, p)
    new_order = order.copy()
    new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
    ms_after = _makespan_fast(new_order, p)
    return ms_after - ms_before


def _local_search(order, p, max_iter=5000, time_limit=30):
    """First-improvement local search with adjacent swaps."""
    n_jobs, n_machines = p.shape
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan_fast(best_order, p)
    start_time = time.time()
    
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        if time.time() - start_time > time_limit:
            break
        for i in range(n_jobs - 1):
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan_fast(new_order, p)
            if new_ms < best_ms - 1e-10:
                best_order = new_order
                best_ms = new_ms
                improved = True
                break  # Restart scan from beginning after improvement
    return best_order


def _simulated_annealing(order, p, time_limit=20):
    """Simulated annealing for flow shop problem."""
    n_jobs, n_machines = p.shape
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan_fast(current_order, p)
    best_order = list(current_order)
    best_ms = current_ms
    
    start_time = time.time()
    initial_temp = max(1.0, current_ms * 0.1)
    cooling_rate = 0.999
    
    temp = initial_temp
    while time.time() - start_time < time_limit and temp > 1e-6:
        # Generate neighbor by random adjacent swap
        if n_jobs > 1:
            i = np.random.randint(0, n_jobs - 1)
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan_fast(new_order, p)
            delta = new_ms - current_ms
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_order = new_order
                current_ms = new_ms
                if current_ms < best_ms - 1e-10:
                    best_order = list(current_order)
                    best_ms = current_ms
        temp *= cooling_rate
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH initialization followed by local search and simulated annealing.
    """
    p = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = p.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Step 1: NEH initialization
    neh_order = _neh_initialization(p)
    neh_ms = _makespan_fast(neh_order, p)
    
    # Also try simple SPT (shortest processing time) and LPT (longest processing time)
    spt_order = list(np.argsort(p.sum(axis=1)))
    spt_ms = _makespan_fast(spt_order, p)
    
    lpt_order = list(np.argsort(-p.sum(axis=1)))
    lpt_ms = _makespan_fast(lpt_order, p)
    
    # Pick best initial solution
    best_order = min([(neh_order, neh_ms), (spt_order, spt_ms), (lpt_order, lpt_ms)], key=lambda x: x[1])[0]
    best_ms = _makespan_fast(best_order, p)
    
    # Step 2: Local search
    if time.time() < deadline:
        ls_order = _local_search(best_order, p, max_iter=20000, time_limit=30)
        ls_ms = _makespan_fast(ls_order, p)
        if ls_ms < best_ms - 1e-10:
            best_order = ls_order
            best_ms = ls_ms
    
    # Step 3: Simulated annealing with multiple restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run SA from best solution
        sa_order = _simulated_annealing(best_order, p, time_limit=min(20, remaining_time * 0.4))
        sa_ms = _makespan_fast(sa_order, p)
        if sa_ms < best_ms - 1e-10:
            best_order = sa_order
            best_ms = sa_ms
        
        # Try SA from different starting points if time remains
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            # Start SA from NEH
            sa2_order = _simulated_annealing(neh_order, p, time_limit=min(15, remaining_time * 0.4))
            sa2_ms = _makespan_fast(sa2_order, p)
            if sa2_ms < best_ms - 1e-10:
                best_order = sa2_order
                best_ms = sa2_ms
            
            remaining_time = deadline - time.time()
            if remaining_time > 8:
                # Start SA from SPT
                sa3_order = _simulated_annealing(spt_order, p, time_limit=min(10, remaining_time * 0.4))
                sa3_ms = _makespan_fast(sa3_order, p)
                if sa3_ms < best_ms - 1e-10:
                    best_order = sa3_order
                    best_ms = sa3_ms
    
    # Final safety check
    if time.time() > deadline:
        # Return best found so far
        pass
    
    return list(best_order)
# EVOLVE-BLOCK-END
