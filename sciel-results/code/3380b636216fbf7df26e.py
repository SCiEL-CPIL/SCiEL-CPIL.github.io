"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list or array of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0

    # Use a vectorized-ish approach for speed
    # C[m] is the completion time of the last processed job on machine m
    # For each job j in order:
    #   C[0] += P[j, 0]
    #   C[m] = max(C[m], C[m-1]) + P[j, m]
    
    # We'll do it in a loop but with numpy operations where possible
    # Actually, for clarity and reasonable speed, let's do a standard loop
    # but optimize the inner part.
    
    # Let's try a different approach: maintain completion times per machine
    # C is an array of size n_machines
    C = np.zeros(n_machines)
    
    for j in order:
        # Machine 0
        C[0] += processing_times[j, 0]
        if n_machines > 1:
            # For machines 1..n_machines-1
            # C[m] = max(C[m], C[m-1]) + P[j, m]
            # This is sequential, so we can't fully vectorize without a scan
            # But we can do it in a loop
            for m in range(1, n_machines):
                C[m] = max(C[m], C[m-1]) + processing_times[j, m]
    
    return C[-1]


def _makespan_fast(order, processing_times):
    """
    Faster makespan calculation using a different formulation.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    # Extract the processing times for the jobs in the given order
    # P_ordered has shape (n_jobs, n_machines)
    P_ordered = processing_times[order]
    
    # C[m] after processing all jobs up to index i
    # We want C_{n-1}[n_machines-1]
    
    # Let's use the recurrence:
    # C[i][0] = C[i-1][0] + P[i][0]
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[i][m]
    
    # We can maintain a single array C of size n_machines
    C = np.zeros(n_machines)
    
    for i in range(len(P_ordered)):
        p = P_ordered[i]
        C[0] += p[0]
        if n_machines > 1:
            # C[m] = max(C[m], C[m-1]) + p[m]
            # We need to do this sequentially
            for m in range(1, n_machines):
                C[m] = max(C[m], C[m-1]) + p[m]
    
    return C[-1]


def _neh_heuristic(processing_times):
    """
    NEH heuristic for flow shop scheduling.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Calculate total processing time for each job
    total_pt = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    order = np.argsort(-total_pt)
    
    # Insert each job into the best position in the current sequence
    seq = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(seq) + 1):
            # Try inserting job at position pos
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_fast(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return np.array(seq)


def _local_search(order, processing_times, max_iterations=1000, time_limit=None):
    """
    Local search using 2-opt and insertion moves.
    """
    current_order = np.array(order)
    current_ms = _makespan_fast(current_order, processing_times)
    best_order = current_order.copy()
    best_ms = current_ms
    
    n_jobs = len(current_order)
    if n_jobs < 2:
        return best_order, best_ms
    
    start_time = time.time()
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Try all 2-opt moves
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Reverse the segment from i to j
                new_order = current_order.copy()
                new_order[i:j+1] = current_order[i:j+1][::-1]
                new_ms = _makespan_fast(new_order, processing_times)
                
                if new_ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_order = current_order.copy()
        
        # Try all insertion moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j or abs(i - j) == 1:
                    continue
                # Move job at position i to position j
                new_order = list(current_order)
                job = new_order.pop(i)
                # Adjust j if i < j
                if i < j:
                    new_order.insert(j - 1, job)
                else:
                    new_order.insert(j, job)
                new_order = np.array(new_order)
                
                new_ms = _makespan_fast(new_order, processing_times)
                
                if new_ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_order = current_order.copy()
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Main function to order jobs for flow shop scheduling.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    start_time = time.time()
    time_limit = 145.0  # Leave 5 seconds margin for the 150s limit
    
    # Start with NEH heuristic
    best_order = _neh_heuristic(processing_times)
    best_ms = _makespan_fast(best_order, processing_times)
    
    # Also try simple heuristics
    # Longest total processing time first
    order_lpt = np.argsort(-processing_times.sum(axis=1))
    ms_lpt = _makespan_fast(order_lpt, processing_times)
    if ms_lpt < best_ms:
        best_ms = ms_lpt
        best_order = order_lpt
    
    # Shortest total processing time first
    order_spt = np.argsort(processing_times.sum(axis=1))
    ms_spt = _makespan_fast(order_spt, processing_times)
    if ms_spt < best_ms:
        best_ms = ms_spt
        best_order = order_spt
    
    # Run local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        ls_order, ls_ms = _local_search(best_order, processing_times, 
                                         max_iterations=500, 
                                         time_limit=remaining_time - 2.0)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # Try multiple random restarts with local search if time permits
    remaining_time = time_limit - (time.time() - start_time)
    max_restarts = 10
    for restart in range(max_restarts):
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        # Generate a random order
        random_order = np.random.permutation(n_jobs)
        
        # Run local search on the random order
        ls_order, ls_ms = _local_search(random_order, processing_times,
                                         max_iterations=200,
                                         time_limit=2.0)
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    return list(best_order)
# EVOLVE-BLOCK-END
