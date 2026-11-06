"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    Optimized for speed using local variables.
    """
    if not order:
        return 0.0
        
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
        
    # Use lists for speed in the inner loop
    curr_completion = [0.0] * n_machines
    
    for j in order:
        p = processing_times[j]
        curr_completion[0] += p[0]
        for m in range(1, n_machines):
            prev_m_comp = curr_completion[m-1]
            curr_m_comp = curr_completion[m]
            if prev_m_comp > curr_m_comp:
                curr_completion[m] = prev_m_comp + p[m]
            else:
                curr_completion[m] = curr_m_comp + p[m]
                
    return curr_completion[-1]


def _neh_heuristic(processing_times, seed=None):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    Includes optional randomization for diversity.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    # If seed is provided, add small noise to break ties and create diversity
    if seed is not None:
        rng = np.random.RandomState(seed)
        # Small perturbation relative to the range
        range_sums = job_sums.max() - job_sums.min()
        if range_sums < 1e-9:
            range_sums = 1.0
        noise = rng.rand(n_jobs) * 0.05 * range_sums
        job_sums = job_sums + noise
    
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _local_search_insertion(order, processing_times, deadline):
    """
    Perform insertion-based local search.
    Uses First Improvement strategy for speed, restarting passes on improvement.
    """
    n = len(order)
    if n == 0:
        return order, 0.0
    
    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    
    # Convert processing times to list of lists for faster indexing if needed? 
    # Usually numpy row access is fine, but let's stick to the original for consistency unless profiling says otherwise.
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return current_order, current_ms
            
            job = current_order[i]
            
            # Try inserting at every position except the current one (i)
            # We can skip i+1 as well? No, i+1 is a valid distinct move (swap with next).
            # Actually, inserting at i is the identity. Inserting at i+1 is a swap.
            # Standard insertion move: remove from i, insert at k.
            # If k == i, no change.
            
            best_pos = i
            best_ms = current_ms
            
            for pos in range(n):
                if pos == i:
                    continue
                
                # Construct new sequence
                # To avoid expensive list slicing every time, we can simulate or just slice.
                # Slicing is O(n). Total O(n^3) per pass.
                # For n < 100, this is fine.
                
                if pos < i:
                    new_seq = current_order[:pos] + [job] + current_order[pos:i] + current_order[i+1:]
                else:
                    # pos > i
                    new_seq = current_order[:i] + current_order[i+1:pos+1] + [job] + current_order[pos+1:]
                
                ms = _makespan(new_seq, processing_times)
                
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_pos != i:
                # Apply the best move found in this pass for job i
                # Reconstruct the sequence
                job = current_order[i]
                if best_pos < i:
                    current_order = current_order[:best_pos] + [job] + current_order[best_pos:i] + current_order[i+1:]
                else:
                    current_order = current_order[:i] + current_order[i+1:best_pos+1] + [job] + current_order[best_pos+1:]
                
                current_ms = best_ms
                improved = True
                # First Improvement: Restart the outer loop to re-evaluate the new sequence from the beginning
                break
                
        if not improved:
            break
            
    return current_order, current_ms


def _simulated_annealing(order, processing_times, deadline, initial_temp=1000.0, cooling=0.999):
    """
    Perform Simulated Annealing to escape local minima.
    Uses adjacent swaps primarily.
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
        # Generate a random neighbor
        # 80% adjacent swap, 20% random swap
        if random.random() < 0.8:
            i = random.randint(0, n - 2)
            j = i + 1
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
        # Perform swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Calculate new makespan
        new_ms = _makespan(current_order, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept better solution
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        else:
            # Accept worse solution with probability exp(-delta / temp)
            if random.random() < np.exp(-delta / temp):
                current_ms = new_ms
            else:
                # Reject: swap back
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        temp *= cooling
        
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    Combines NEH with restarts, insertion local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy: Multiple restarts.
    # For small N, we can do many restarts.
    # For large N, local search is slower, so fewer restarts.
    
    # Determine number of restarts based on time and problem size
    # Let's aim for a dynamic number of restarts.
    
    # First, do one full run to establish a baseline and estimate time per run.
    # Or just run a fixed number if time allows.
    
    # Let's try up to 10 restarts, but stop if time is up.
    max_restarts = 10
    
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Use different seeds for NEH to get diverse starts
        # Seed 0 is deterministic NEH. Others are randomized.
        if r == 0:
            initial_order = _neh_heuristic(processing_times, seed=None)
        else:
            initial_order = _neh_heuristic(processing_times, seed=r * 12345 + 99)
        
        initial_ms = _makespan(initial_order, processing_times)
        
        # If this initial is already better, update best
        if initial_ms < best_ms:
            best_ms = initial_ms
            best_order = initial_order[:]
        
        # Allocate time for local search and SA for this restart
        # We want to leave some time for subsequent restarts if possible.
        # Simple heuristic: give each restart an equal share of the remaining time?
        # Or just run until a local optimum is found, then do a short SA.
        
        # Let's do:
        # 1. Local Search (Insertion) - First Improvement, run for a limited time or until convergence.
        # 2. Simulated Annealing - Run for a limited time to escape local min.
        
        # Time budgeting:
        # If we have 10 restarts, and 115s total, that's ~11s per restart.
        # Let's say 5s for LS, 5s for SA, 1s overhead.
        
        ls_deadline = time.time() + 5.0
        if ls_deadline > deadline - 10.0: # Ensure we have time for future restarts
             ls_deadline = deadline - 10.0 - (max_restarts - r - 1) * 10.0 # Rough estimate
             if ls_deadline < time.time() + 1.0:
                 ls_deadline = time.time() + 1.0 # At least 1s
        
        ls_order, ls_ms = _local_search_insertion(initial_order, processing_times, ls_deadline)
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order[:]
        
        # Now SA
        sa_deadline = time.time() + 5.0
        if sa_deadline > deadline - 5.0:
             sa_deadline = deadline - 5.0 - (max_restarts - r - 1) * 10.0
             if sa_deadline < time.time() + 1.0:
                 sa_deadline = time.time() + 1.0
        
        sa_order, sa_ms = _simulated_annealing(ls_order, processing_times, sa_deadline)
        
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order[:]
            
    # If we still have time, do a final long SA on the best solution found
    if time.time() < deadline - 2.0 and best_order is not None:
        final_sa_deadline = deadline - 1.0
        final_order, final_ms = _simulated_annealing(best_order, processing_times, final_sa_deadline, initial_temp=500.0, cooling=0.9995)
        if final_ms < best_ms:
            best_ms = final_ms
            best_order = final_order[:]

    if best_order is None:
        # Fallback
        best_order = _neh_heuristic(processing_times)
        
    return best_order

# EVOLVE-BLOCK-END
