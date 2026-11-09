"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """
    Optimized makespan calculation for flow shop.
    """
    n_jobs, n_machines = processing_times.shape
    # completion[m] is the time when machine m becomes free
    completion = [0.0] * n_machines
    
    for job in perm:
        p = processing_times[job]
        prev_finish = 0.0
        for m in range(n_machines):
            # Start time is max(machine free time, job finished previous machine)
            start = completion[m] if completion[m] > prev_finish else prev_finish
            finish = start + p[m]
            completion[m] = finish
            prev_finish = finish
            
    return completion[-1]


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Parameters
    # Scale temperature based on problem magnitude
    avg_p = np.mean(processing_times)
    if avg_p == 0:
        avg_p = 1.0
        
    # Number of restarts
    # More restarts for small problems to explore more
    if n_jobs <= 20:
        num_restarts = 20
        iters_per_restart = 20000
    elif n_jobs <= 50:
        num_restarts = 10
        iters_per_restart = 10000
    else:
        num_restarts = 5
        iters_per_restart = 5000
        
    # Global best
    best_perm = list(range(n_jobs))
    random.shuffle(best_perm)
    best_cost = _makespan(best_perm, processing_times)
    
    for _ in range(num_restarts):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random solution
        current_perm = list(range(n_jobs))
        random.shuffle(current_perm)
        current_cost = _makespan(current_perm, processing_times)
        
        # If this random start is already better, update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = current_perm[:]
            
        # SA Cooling
        # T_start should allow significant exploration
        # T_end should be low to converge
        T_start = avg_p * 5.0
        T_end = avg_p * 0.01
        
        if iters_per_restart > 0:
            alpha = (T_end / T_start) ** (1.0 / iters_per_restart)
        else:
            alpha = 1.0
            
        T = T_start
        
        for _ in range(iters_per_restart):
            # Check time frequently
            if time.time() - start_time > time_limit:
                break
                
            # Move: Swap two random jobs
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
                
            # Apply move
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            new_cost = _makespan(current_perm, processing_times)
            
            delta = new_cost - current_cost
            
            if delta < 0:
                # Accept
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_perm = current_perm[:]
            else:
                # Reject or accept with probability
                if T > 0:
                    # Avoid overflow in exp
                    if delta / T < 700:
                        prob = np.exp(-delta / T)
                    else:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
            T *= alpha
            
    return best_perm
# EVOLVE-BLOCK-END
