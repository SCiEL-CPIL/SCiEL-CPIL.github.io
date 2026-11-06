import numpy as np
import time as time_module
from collections import defaultdict

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 145.0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with min duration? Or just return index?
        # Returning (index, machine) is allowed.
        # If only one op, pick the machine that minimizes immediate load?
        # For a single op, any machine is fine for LB, but let's pick min duration to keep it short.
        min_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, min_machine)

    # 1. Build the "Future Work" map
    # job_machine_work[job][machine] = total duration remaining for job 'job' on machine 'machine'
    # job_remaining_work[job] = total duration remaining for job 'job'
    
    job_machine_work = defaultdict(lambda: defaultdict(float))
    job_remaining_work = defaultdict(float)
    
    for op in ready_operations:
        job = op["job"]
        # Current op options
        # Note: The op is "ready", meaning it hasn't started.
        # So its duration is part of the remaining work.
        for m, d in op["machine_options"].items():
            job_machine_work[job][m] += d
            job_remaining_work[job] += d
            
        # Future ops
        rem_ops = op.get("remaining_ops", [])
        if isinstance(rem_ops, list):
            for next_op in rem_ops:
                # Assume next_op has similar structure
                # If it has 'machine_options', use it
                opts = next_op.get("machine_options", {})
                if opts:
                    for m, d in opts.items():
                        job_machine_work[job][m] += d
                        job_remaining_work[job] += d
                else:
                    # Fallback if structure is different? 
                    # Try to extract duration
                    dur = next_op.get("duration", 0)
                    # If no machine specified, we can't add to specific machine map accurately
                    # But we can add to job_remaining_work
                    job_remaining_work[job] += dur

    # Get list of all machines involved (assume 5, but derive from data to be safe)
    all_machines = set()
    for job in job_machine_work:
        all_machines.update(job_machine_work[job].keys())
    if not all_machines:
        return 0 # Fallback
    
    all_machines = list(all_machines)
    n_machines = len(all_machines)
    
    # Pre-compute initial totals for speed? 
    # We will modify and restore, so we need a snapshot or just recalculate.
    # Recalculation is O(J*M). J=10, M=5. Very fast.
    
    best_score = float('inf')
    best_idx = 0
    best_machine = None
    best_lpt_tiebreaker = -1.0
    best_rw_tiebreaker = float('inf')

    for i, op in enumerate(ready_operations):
        job = op["job"]
        current_rw = job_remaining_work[job]
        
        for m, d in op["machine_options"].items():
            # Simulate: Assign op to machine m
            # Update job_machine_work[job][m]
            old_m_val = job_machine_work[job][m]
            job_machine_work[job][m] = old_m_val - d
            
            # Update job_remaining_work[job]
            old_j_val = job_remaining_work[job]
            job_remaining_work[job] = old_j_val - d
            
            # Calculate LB
            # 1. Max Job Remaining Work
            max_job_rw = 0.0
            for j in job_remaining_work:
                if job_remaining_work[j] > max_job_rw:
                    max_job_rw = job_remaining_work[j]
            
            # 2. Max Machine Load
            max_machine_load = 0.0
            for m_k in all_machines:
                load = 0.0
                for j in job_machine_work:
                    load += job_machine_work[j][m_k]
                if load > max_machine_load:
                    max_machine_load = load
            
            c_max = max(max_job_rw, max_machine_load)
            
            # Restore
            job_machine_work[job][m] = old_m_val
            job_remaining_work[job] = old_j_val
            
            # Score
            # Primary: Minimize c_max
            # Secondary: Tie-breaker 1: Maximize duration (LPT)
            # Secondary: Tie-breaker 2: Minimize remaining work of the job (finish it)
            
            if c_max < best_score - 1e-9:
                best_score = c_max
                best_idx = i
                best_machine = m
                best_lpt_tiebreaker = d
                best_rw_tiebreaker = job_remaining_work[job]
            elif abs(c_max - best_score) < 1e-9:
                # Tie-break
                if d > best_lpt_tiebreaker + 1e-9:
                    best_idx = i
                    best_machine = m
                    best_lpt_tiebreaker = d
                    best_rw_tiebreaker = job_remaining_work[job]
                elif abs(d - best_lpt_tiebreaker) < 1e-9:
                    if job_remaining_work[job] < best_rw_tiebreaker - 1e-9:
                        best_idx = i
                        best_machine = m
                        best_lpt_tiebreaker = d
                        best_rw_tiebreaker = job_remaining_work[job]
                        
        if time_module.time() > deadline:
            break

    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
