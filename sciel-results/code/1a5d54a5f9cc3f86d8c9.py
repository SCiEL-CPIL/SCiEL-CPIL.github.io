"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    if not order:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # Vectorized makespan calculation
    # Extract processing times for the jobs in the given order
    # processing_times[order] gives an array of shape (n_jobs, n_machines)
    p = processing_times[order]
    
    # If only one machine, makespan is sum of processing times
    if n_machines == 1:
        return float(p[:, 0].sum())
        
    # For multiple machines:
    # C[0] is cumulative sum of p[:, 0]
    # C[m] = max(C[m-1], C[m]_prev) + p[:, m]
    # We can compute this iteratively using numpy arrays.
    
    # Initialize completion time array for all jobs on all machines
    # We don't need to store all, just the current machine's completion times
    # C_m[j] = max(C_{m-1}[j], C_m[j-1]) + p[j, m]
    
    # C_0 = cumsum(p[:, 0])
    C = np.cumsum(p[:, 0])
    
    for m in range(1, n_machines):
        # C_m[j] = max(C_{m-1}[j], C_m[j-1]) + p[j, m]
        # C_m[j-1] is the previous element in the current array C (which represents C_m for j-1)
        # But we are updating C in place.
        # Let's do it carefully.
        # new_C[0] = C[0] + p[0, m]
        # new_C[j] = max(C[j], new_C[j-1]) + p[j, m]
        
        # We can use a loop for the vectorized update because the dependency is sequential (j depends on j-1)
        # However, n_jobs is typically small enough (e.g., < 100) that a python loop over jobs is fast,
        # OR we can use a trick with cumulative max if we reformulate.
        
        # Actually, for flow shop, C_m[j] = max(C_{m-1}[j], C_m[j-1]) + p[j,m]
        # This is equivalent to:
        # C_m = np.maximum.accumulate(C_{m-1} + cumsum(p[:, m])) ? No, that's not quite right.
        
        # Let's stick to the efficient loop for the machine dimension, but vectorize the job dimension?
        # No, the job dimension has sequential dependency.
        
        # Standard efficient numpy approach for small n_jobs:
        # Loop over jobs is O(N), loop over machines is O(M).
        # Current implementation was O(N*M) in Python.
        # Let's try to keep it in Python but with minimal overhead, or use a specific numpy trick.
        
        # Trick:
        # C_m[j] = max(C_{m-1}[j], C_m[j-1]) + p[j, m]
        # Let A[j] = C_{m-1}[j] + p[j, m]
        # Then C_m[0] = A[0]
        # C_m[j] = max(A[j], C_m[j-1])
        # So C_m = np.maximum.accumulate(A)
        
        A = C + p[:, m]
        C = np.maximum.accumulate(A)
        
    return float(C[-1])


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            # Optimization: Instead of creating new list and calling _makespan,
            # we could do incremental, but for NEH initialization, n is usually small enough 
            # or we can just use the fast _makespan.
            # To be safe and fast, we construct the list.
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _local_search(order, processing_times, deadline):
    """
    Perform Insertion-based local search.
    Returns improved order and makespan.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try insertion moves: remove job at i, insert at j
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Remove job at i
            job = current_order[i]
            # Create a temporary order without this job
            # We can simulate this by creating a new list, but that's O(N) copy.
            # Since we are doing O(N^2) moves, O(N) copy per move is O(N^3) total.
            # For N=100, 10^6 * 100 = 10^8 ops, might be too slow in Python.
            # Let's optimize the makespan calculation or the move simulation.
            
            # Alternative: Just do the list manipulation and call _makespan.
            # _makespan is now vectorized and fast.
            
            rest_order = current_order[:i] + current_order[i+1:]
            
            for j in range(len(rest_order) + 1):
                if time.time() > deadline:
                    break
                
                # Insert job at position j in rest_order
                new_order = rest_order[:j] + [job] + rest_order[j:]
                
                new_ms = _makespan(new_order, processing_times)
                
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    # Break inner loop to restart scan with new order
                    # This ensures we don't use a stale 'rest_order'
                    break
            
            # If we improved, we break the outer loop to restart the whole pass
            if improved:
                break
                
    return current_order, current_ms


def _simulated_annealing(order, processing_times, deadline, initial_temp=1000.0, cooling=0.999):
    """
    Perform Simulated Annealing to escape local minima.
    Uses random insertion moves for better exploration.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    best_order = current_order[:]
    best_ms = current_ms
    
    temp = initial_temp
    
    while time.time() < deadline and temp > 1e-6:
        # Generate a random neighbor using insertion
        i = random.randint(0, n - 1)
        
        # Remove job at i
        job = current_order[i]
        rest_order = current_order[:i] + current_order[i+1:]
        
        # Insert at random position j
        j = random.randint(0, len(rest_order))
        
        new_order = rest_order[:j] + [job] + rest_order[j:]
        
        new_ms = _makespan(new_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept better solution
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        else:
            # Accept worse solution with probability exp(-delta / temp)
            # Clamp delta to avoid underflow/overflow in exp
            if delta > 1000 * temp:
                prob = 0.0
            else:
                prob = np.exp(-delta / temp)
            
            if random.random() < prob:
                current_order = new_order
                current_ms = new_ms
                
        temp *= cooling
        
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local Search
    # Allocate 60% of time to local search, 40% to SA
    # Insertion LS is powerful but can be slow if N is large.
    # Vectorized makespan helps a lot.
    ls_deadline = start_time + (deadline - start_time) * 0.6
    
    current_order, current_makespan = _local_search(current_order, processing_times, ls_deadline)
    
    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_order = current_order[:]
        
    # 3. Simulated Annealing
    if time.time() < deadline:
        # Use a higher initial temperature for SA to explore more
        # Scale temp based on typical makespan magnitude? 
        # Let's estimate typical makespan to set temp
        avg_pt = np.mean(processing_times)
        est_ms = n_jobs * n_machines * avg_pt
        initial_temp = max(10.0, est_ms * 0.1)
        
        sa_order, sa_ms = _simulated_annealing(current_order, processing_times, deadline, initial_temp=initial_temp, cooling=0.999)
        
        if sa_ms < best_makespan:
            best_makespan = sa_ms
            best_order = sa_order[:]
            
    # 4. Final Polish: Run a quick local search on the best solution found
    # If we have time left, try to polish the best solution found (which might be from SA)
    if time.time() < deadline - 5:
        polish_deadline = deadline - 5
        polished_order, polished_ms = _local_search(best_order, processing_times, polish_deadline)
        if polished_ms < best_makespan:
            best_makespan = polished_ms
            best_order = polished_order[:]

    return best_order

# EVOLVE-BLOCK-END
