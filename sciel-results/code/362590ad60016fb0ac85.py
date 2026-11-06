"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    
    This approach uses a stochastic search to escape local optima, unlike the 
    deterministic NEH + Local Search approach.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Initialize with a random permutation
    current_seq = list(range(n_jobs))
    random.shuffle(current_seq)
    
    # Calculate initial makespan
    current_makespan = _calc_makespan(processing_times, current_seq, n_machines)
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # Simulated Annealing Parameters
    # Start temperature high enough to accept most moves
    start_temp = 100.0 
    # End temperature low to converge
    end_temp = 0.1
    # Number of iterations
    # Adjust based on problem size to fit within time limit
    # N=10: ~100k iters, N=100: ~10k iters
    max_iterations = 200000 if n_jobs < 20 else (50000 if n_jobs < 50 else 10000)
    
    start_time = time.time()
    time_limit = 100.0  # seconds
    
    for i in range(max_iterations):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
        # Cooling schedule: Linear or Exponential
        # Linear is often more stable for this size
        temp = start_temp - (start_temp - end_temp) * (i / max_iterations)
        
        # Generate a neighbor using insertion move
        new_seq = current_seq.copy()
        
        # Pick a random job to move
        job_idx = random.randint(0, n_jobs - 1)
        job_to_move = new_seq[job_idx]
        
        # Remove it
        new_seq.pop(job_idx)
        
        # Insert at a random position
        insert_idx = random.randint(0, n_jobs - 1)
        new_seq.insert(insert_idx, job_to_move)
        
        # Calculate new makespan
        new_makespan = _calc_makespan(processing_times, new_seq, n_machines)
        
        # Acceptance criterion
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Better solution
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = current_seq.copy()
        else:
            # Worse solution, accept with probability exp(-delta / temp)
            if temp > 0:
                probability = np.exp(-delta / temp)
                if random.random() < probability:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    
    return best_seq


def _calc_makespan(processing_times, seq, n_machines):
    """Calculate the makespan for a given job sequence."""
    # completion_times[m] is the completion time on machine m for the current job
    completion_times = [0.0] * n_machines
    
    for job_idx in seq:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += processing_times[job_idx, m]
            else:
                # max(completion_times[m], completion_times[m-1]) + processing_times[job_idx, m]
                prev_max = completion_times[m] if completion_times[m] > completion_times[m-1] else completion_times[m-1]
                completion_times[m] = prev_max + processing_times[job_idx, m]
    
    return completion_times[-1]
# EVOLVE-BLOCK-END
