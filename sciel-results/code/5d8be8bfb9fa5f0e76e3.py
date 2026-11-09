"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on the flow shop."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    # completion_times[m] tracks the completion time on machine m
    completion_times = np.zeros(n_machines)
    for job_idx in order:
        for m in range(n_machines):
            completion_times[m] = max(completion_times[m], completion_times[m-1] if m > 0 else 0.0) + processing_times[job_idx, m]
    return completion_times[-1]


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Build completion matrix C[j][m] = completion time of job j on machine m
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return C[n_jobs-1, n_machines-1]


def _neh_order(processing_times):
    """NEH heuristic: order jobs by decreasing total processing time, insert greedily."""
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    return _construct_insertion_order(processing_times, order)


def _construct_insertion_order(processing_times, order):
    """Given an initial order, greedily insert jobs one by one to minimize makespan."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return []
    
    # Start with the first job
    current_order = [order[0]]
    
    for job in order[1:]:
        # Try inserting job at each possible position in current_order
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(current_order) + 1):
            candidate = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan_fast(candidate, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_order = current_order[:best_pos] + [job] + current_order[best_pos:]
    
    return current_order


def _local_search(processing_times, order, time_limit=5.0):
    """Local search using adjacent swaps and insertion moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan_fast(list(order), processing_times)
    
    current_order = list(order)
    current_ms = _makespan_fast(current_order, processing_times)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Adjacent swaps
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            candidate = current_order.copy()
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan_fast(candidate, processing_times)
            if ms < current_ms:
                current_order = candidate
                current_ms = ms
                improved = True
        
        # Insertion moves
        for i in range(n_jobs):
            if (time.time() - start_time) >= time_limit:
                break
            job = current_order[i]
            remaining = [j for j in current_order if j != job]
            for j in range(len(remaining) + 1):
                if (time.time() - start_time) >= time_limit:
                    break
                candidate = remaining[:j] + [job] + remaining[j:]
                ms = _makespan_fast(candidate, processing_times)
                if ms < current_ms:
                    current_order = candidate
                    current_ms = ms
                    improved = True
                    break
    
    return current_order, current_ms


def _simulated_annealing(processing_times, order, time_limit=10.0):
    """Simulated Annealing with 2-opt block reversal and insertion moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan_fast(list(order), processing_times)
    
    current_order = list(order)
    current_ms = _makespan_fast(current_order, processing_times)
    best_order = list(current_order)
    best_ms = current_ms
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Temperature schedule
    T0 = max(1.0, current_ms * 0.01)
    Tend = T0 * 0.001
    n_iterations = 2000
    cooling_rate = (Tend / T0) ** (1.0 / max(1, n_iterations))
    T = T0
    
    rng = np.random.default_rng(12345)
    
    for it in range(n_iterations):
        if time.time() >= deadline:
            break
        
        # Generate neighbor: 2-opt block reversal or insertion
        if rng.random() < 0.5 and n_jobs >= 3:
            # 2-opt block reversal
            i = rng.integers(0, n_jobs - 1)
            j = rng.integers(i + 1, n_jobs)
            candidate = current_order[:i] + list(reversed(current_order[i:j+1])) + current_order[j+1:]
        else:
            # Insertion move
            i = rng.integers(0, n_jobs)
            j = rng.integers(0, n_jobs)
            if i == j:
                candidate = list(current_order)
            else:
                job = current_order[i]
                remaining = current_order[:i] + current_order[i+1:]
                insert_pos = j if j < i else j - 1
                candidate = remaining[:insert_pos] + [job] + remaining[insert_pos:]
        
        if candidate == current_order:
            T *= cooling_rate
            continue
        
        candidate_ms = _makespan_fast(candidate, processing_times)
        delta = candidate_ms - current_ms
        
        # Acceptance criterion
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_order = candidate
            current_ms = candidate_ms
            
            if current_ms < best_ms:
                best_order = list(current_order)
                best_ms = current_ms
        
        T *= cooling_rate
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial heuristic: NEH with insertion
    best_order = _neh_order(processing_times)
    best_ms = _makespan_fast(best_order, processing_times)
    
    # Local search to polish
    ls_order, ls_ms = _local_search(processing_times, best_order, time_limit=5.0)
    if ls_ms < best_ms:
        best_order = ls_order
        best_ms = ls_ms
    
    # Simulated Annealing to escape local optima
    elapsed = time.time() - start_time
    sa_time = min(30.0, deadline - start_time - 5.0)
    if sa_time > 2.0:
        sa_order, sa_ms = _simulated_annealing(processing_times, best_order, time_limit=sa_time)
        if sa_ms < best_ms:
            best_order = sa_order
            best_ms = sa_ms
        
        # Final local search polish after SA
        elapsed = time.time() - start_time
        if elapsed < deadline - 3.0:
            final_order, final_ms = _local_search(processing_times, best_order, time_limit=min(10.0, deadline - start_time - 2.0))
            if final_ms < best_ms:
                best_order = final_order
                best_ms = final_ms
    
    # If time remains, try a few more SA restarts from perturbed orders
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0 and n_jobs > 5:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if time.time() >= deadline - 3.0:
                break
            # Perturb: random 2-opt reversal
            perturbed = best_order.copy()
            i, j = sorted(rng.integers(0, n_jobs, size=2).tolist())
            if i < j:
                perturbed[i:j+1] = list(reversed(perturbed[i:j+1]))
            
            sa2_time = min(10.0, deadline - start_time - 3.0)
            if sa2_time > 1.0:
                sa2_order, sa2_ms = _simulated_annealing(processing_times, perturbed, time_limit=sa2_time)
                if sa2_ms < best_ms:
                    best_order = sa2_order
                    best_ms = sa2_ms
                    # Quick local search polish
                    if time.time() < deadline - 2.0:
                        ls2_order, ls2_ms = _local_search(processing_times, best_order, time_limit=min(5.0, deadline - start_time - 1.0))
                        if ls2_ms < best_ms:
                            best_order = ls2_order
                            best_ms = ls2_ms
    
    return list(best_order)
# EVOLVE-BLOCK-END
