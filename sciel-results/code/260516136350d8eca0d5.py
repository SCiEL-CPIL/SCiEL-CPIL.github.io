"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    Optimized for speed using Python lists and local variables.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    
    # completion[m] = completion time of the last processed job on machine m
    # We don't need to store the full matrix, just the last job's completion on each machine
    # But we also need the completion time of the *current* job on the *previous* machine.
    # So we track:
    # 1. prev_job_completion_on_machines: array of length n_machines
    # 2. current_job_completion_on_prev_machine: scalar
    
    # Actually, the standard DP is:
    # C[j, m] = max(C[j, m-1], C[j-1, m]) + P[j, m]
    # We can compute this row by row (job by job).
    
    # To be very fast, we avoid numpy overhead inside the loop if possible, 
    # but numpy is faster for the vector ops if we can do them.
    # However, the dependency is sequential.
    
    # Let's use a list of lists or just track the necessary values.
    # C_prev_machine[j] = completion time of job j on machine m-1
    # C_prev_job_on_m[m] = completion time of previous job on machine m
    
    # We iterate through jobs in perm.
    # For each job, we compute its completion times on all machines.
    
    # State:
    # last_job_completion_on_machines: array of size n_machines, initially 0
    # current_job_completion: scalar, initially 0 (completion on machine 0 before processing? No, starts at 0)
    
    # Let's trace:
    # Job 0:
    #   M0: max(0, 0) + P[0,0] = P[0,0]
    #   M1: max(C[0,0], 0) + P[0,1]
    #   ...
    # Job 1:
    #   M0: max(0, C[0,0]) + P[1,0]  (Here C[0,0] is completion of prev job on M0)
    #   M1: max(C[1,0], C[0,1]) + P[1,1]
    
    # So we need:
    # 1. The completion time of the *previous job* on each machine.
    # 2. The completion time of the *current job* on the *previous machine*.
    
    prev_job_completion = np.zeros(n_machines)
    
    for job in perm:
        current_job_completion_on_prev_machine = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                # Start time is max(0, prev_job_completion[0])
                # Since time starts at 0, and prev_job_completion[0] is >= 0
                start = prev_job_completion[0]
                current_job_completion_on_prev_machine = start + p
            else:
                # Start time is max(completion of current job on m-1, completion of prev job on m)
                start = max(current_job_completion_on_prev_machine, prev_job_completion[m])
                current_job_completion_on_prev_machine = start + p
            
            # Update prev_job_completion[m] to be the completion of the current job on this machine
            # Wait, this logic is slightly off. 
            # prev_job_completion[m] should hold the completion time of the job *before* the current one on machine m.
            # After processing the current job on machine m, the completion time for the *next* job on machine m 
            # will be the current job's completion time on machine m.
            # So we update prev_job_completion[m] = current_job_completion_on_prev_machine
            
            prev_job_completion[m] = current_job_completion_on_prev_machine
        
        # The makespan is the completion time of the last job on the last machine.
        # This is stored in prev_job_completion[-1] after the last job is processed.
    
    return prev_job_completion[-1]


def _neh_initialization(processing_times):
    """
    NEH heuristic for initial permutation.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the position that minimizes the incremental makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Step 1: Sort by total processing time
    total_processing = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_processing)
    
    # Step 2: Insert one by one
    current_perm = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm.copy()
            new_perm.insert(pos, job)
            ms = _makespan(new_perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm.insert(best_pos, job)
        
    return current_perm


def _tabu_search(initial_perm, processing_times, time_limit=10.0):
    """
    Tabu search with swaps and insertions.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: dictionary mapping move tuple to expiration time
    # Move types: ('swap', i, j) or ('move', i, j) where i is removed from i and inserted at j
    tabu_list = {}
    tabu_tenure = 10
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50  # Restart or stop if no improvement for a while
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Neighborhood: 
        # 1. Adjacent swaps
        # 2. Moves (remove job i, insert at j)
        
        # To limit computation, we sample moves for large n_jobs
        if n_jobs <= 30:
            # All adjacent swaps
            moves_to_try = []
            for i in range(n_jobs - 1):
                moves_to_try.append(('swap', i, i + 1))
            # Some random moves
            for _ in range(50):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('move', i, j))
        else:
            # Sample moves
            moves_to_try = []
            for _ in range(200):
                if np.random.rand() < 0.5:
                    i = np.random.randint(0, n_jobs - 1)
                    moves_to_try.append(('swap', i, i + 1))
                else:
                    i = np.random.randint(0, n_jobs)
                    j = np.random.randint(0, n_jobs)
                    if i != j:
                        moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('move', i, j)
            
            # Check tabu
            if move_key in tabu_list and tabu_list[move_key] > iteration:
                # Aspiration: accept if it improves best
                # We don't know the makespan yet, so we can't check aspiration here easily without evaluating.
                # We'll just skip tabu moves for now to keep it simple, or evaluate and then check.
                # Let's evaluate first, then check tabu/aspiration.
                pass
            
            # Generate new permutation
            new_perm = current_perm.copy()
            if move[0] == 'swap':
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                # Move job at i to position j
                job_to_move = new_perm.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
            
            new_makespan = _makespan(new_perm, processing_times)
            
            # Check tabu and aspiration
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_perm = new_perm
                best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No improving move found
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            # Perturb: pick a random move
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                new_perm = current_perm.copy()
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
                current_perm = new_perm
                current_makespan = _makespan(current_perm, processing_times)
        else:
            no_improve_count = 0
            # Update tabu list
            tabu_list[best_neighbor_move] = iteration + tabu_tenure
            
            # Clean up tabu list
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
            
            # Accept move
            current_perm = best_neighbor_perm
            current_makespan = best_neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses NEH initialization and tabu search with aspiration.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: NEH
    initial_perm = _neh_initialization(processing_times)
    
    # Refine with tabu search
    # Adjust time limit based on problem size
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 15.0
    elif n_jobs <= 50:
        time_limit = 10.0
    else:
        time_limit = 5.0
    
    final_perm = _tabu_search(initial_perm, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
