import numpy as np
import time as time_module
import random
from collections import defaultdict

# Global cache for machine load estimation if needed, but dispatch is stateless per call
# We rely on the 'ready_operations' list which contains all currently available ops.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 140.0  # 10s safety margin before 150s kill
    
    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute features for each ready operation
    # We need to be fast. Use lists for speed.
    min_durs = np.zeros(n)
    job_ids = np.zeros(n, dtype=np.int32)
    remaining_works = np.zeros(n)
    n_rem_ops = np.zeros(n, dtype=np.int32)
    machine_keys = [None] * n

    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_durs[i] = min(durations) if durations else 0
        job_ids[i] = op["job"]
        remaining_works[i] = op["remaining_work"]
        rem_ops = op.get("remaining_ops", [])
        n_rem_ops[i] = len(rem_ops) if isinstance(rem_ops, list) else 0
        # Store the key of the machine with min duration for fast access later
        if durations:
            min_val = min(durations)
            # Find the machine key with this min duration
            # This is O(M) per op, M is small (5)
            for m, d in op["machine_options"].items():
                if d == min_val:
                    machine_keys[i] = m
                    break
        else:
            machine_keys[i] = -1

    # 1. Static Score for Initial Filtering (Top K)
    # LPT (Longest Processing Time) is generally good for makespan.
    # Add urgency: Jobs with more remaining work or ops should be prioritized.
    # Penalty: If an op has a very long duration, it blocks a machine.
    
    # Normalize features for a robust score
    # Score = LPT * 1.0 + Urgency * 0.5
    urgency = remaining_works + n_rem_ops * 10.0 # Heuristic weight for chain length
    
    static_scores = min_durs * 1.0 + urgency * 0.2
    
    # Get Top K candidates
    K = min(10, n)
    top_indices = np.argsort(static_scores)[::-1][:K]
    
    best_idx = int(top_indices[0])
    best_score = float('inf')
    
    # 2. Lookahead Evaluation
    # For each top candidate, simulate scheduling it.
    # Calculate a "Tightness" metric for the resulting state.
    
    # To calculate Tightness, we need to estimate the remaining work per job and per machine.
    # Current state:
    # Job remaining work: sum of remaining_work for all ops in that job.
    # Machine load: sum of min_durs for all ops that *can* use that machine? 
    # Actually, in open shop, an op MUST be scheduled on one of its machines.
    # A lower bound on makespan is max( max_job_remaining_work, max_machine_load_lower_bound ).
    
    # Let's precompute current job remaining work
    unique_jobs = np.unique(job_ids)
    job_rw = np.zeros(len(unique_jobs))
    job_map = {j: i for i, j in enumerate(unique_jobs)}
    for i in range(n):
        job_rw[job_map[job_ids[i]]] += remaining_works[i]
    
    current_max_job_rw = np.max(job_rw) if len(job_rw) > 0 else 0
    
    # For each candidate, we simulate removing it.
    for idx in top_indices:
        if time_module.time() > deadline:
            break
            
        idx_int = int(idx)
        
        # Simulate: This op is scheduled.
        # It takes min_durs[idx_int] time.
        # The job's remaining work decreases by min_durs[idx_int].
        
        new_job_rw = job_rw.copy()
        j_idx = job_map[job_ids[idx_int]]
        new_job_rw[j_idx] -= min_durs[idx_int]
        
        # Other ops remain.
        # Machine Load Estimation:
        # We estimate the load on each machine by summing the min_durs of all ops 
        # that are forced to use that machine or have it as their best option?
        # A simple proxy: For each machine, sum the min_durs of all ready ops 
        # that include that machine in their options. This overestimates load 
        # because an op only goes to ONE machine.
        # Better proxy: Assign each op to its min-duration machine for the load calc.
        
        # Pre-assignment of ops to machines based on min duration (for load estimation)
        # This is an approximation.
        
        # Let's calculate the load for each machine if we assign every op to its min-dur machine.
        # Machines are 0..4 (assumed 5 machines based on task desc, but keys might be strings/ints)
        # We need to map machine keys to indices.
        # Since we don't know the machine IDs globally, we'll use a local mapping.
        
        # Re-evaluate load for the new state (without the scheduled op)
        # We can pre-compute the "base load" for each machine from the *other* ops.
        
        # Optimization: 
        # Instead of recalculating from scratch, we can observe that only the scheduled op changes.
        # But the "machine load" metric is tricky because the *choice* of machine for other ops might change.
        # However, for a 1-step lookahead, assuming other ops keep their "preferred" (min-dur) machine is a standard greedy approximation.
        
        # Let's compute the load vector for the current state assuming min-dur assignment.
        # We do this once outside the loop if possible? 
        # The set of ready ops changes (one is removed), so loads change.
        
        # Let's just compute the max machine load proxy for the new state.
        # Load_m = sum(min_dur[i] for i in ready_ops if machine_m is in options[i] and min_dur[i] is the min for i)
        
        # To be faster, we can pre-compute the "contribution" of each op to each machine's load (if assigned to min-dur machine).
        # But we don't know the machine indices.
        
        # Let's stick to a simpler, robust metric:
        # Tightness = max( new_max_job_rw, estimated_max_machine_load )
        
        # Estimated Max Machine Load:
        # We approximate machine load by Total_Work / 5.
        # Total_Work_new = sum(remaining_works) - min_durs[idx_int]
        total_work = np.sum(remaining_works)
        new_total_work = total_work - min_durs[idx_int]
        
        # Machine load lower bound is new_total_work / 5
        machine_lb = new_total_work / 5.0
        
        # Job critical path lower bound is new_max_job_rw
        job_lb = np.max(new_job_rw) if len(new_job_rw) > 0 else 0
        
        # Makespan lower bound estimate
        lb = max(job_lb, machine_lb)
        
        # Add a penalty for "unbalanced" jobs? 
        # If one job has much more work than others, it's the bottleneck.
        # The max captures this.
        
        # Tie-breaker: Prefer ops that have fewer options (more constrained) to avoid them waiting?
        # Or prefer ops that finish earlier?
        # Let's use the static score as a tie-breaker.
        
        score = lb - static_scores[idx_int] * 0.1 # Slight preference for high static score if LB is same
        
        if score < best_score:
            best_score = score
            best_idx = idx_int
            
    return best_idx
