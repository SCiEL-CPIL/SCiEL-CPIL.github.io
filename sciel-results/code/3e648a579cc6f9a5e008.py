import numpy as np
import time as time_module
import random
from collections import defaultdict

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 145.0  # 5s safety margin before 150s kill
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    features = []
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        
        # Number of remaining ops in this job
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        
        # Feature: how many machines are options
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "n_rem_ops": n_rem,
            "n_machines": n_machines,
            # LPT feature: prioritize long min-duration ops
            "lpt_score": min_dur,
            # Flexibility: fewer machine options = more constrained
            "constraint": 1.0 / max(n_machines, 1)
        })
    
    # --- Corrected soft_max using log-sum-exp ---
    def soft_max(values, beta=10.0):
        """Proper log-sum-exp approximation to max: (1/beta)*log(sum(exp(beta*x)))"""
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        m = np.max(arr)
        if m <= 0:
            return float(m)
        return float(m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta)
    
    def estimate_makespan(ready_ops, current_time):
        """Estimate makespan: max(job critical paths, machine load balance)."""
        if not ready_ops:
            return current_time
        
        # Per-job remaining work (sum of all ops' remaining work per job)
        job_rw = defaultdict(float)
        total_work = 0.0
        for op in ready_ops:
            job_rw[op["job"]] += op["remaining_work"]
            total_work += op["remaining_work"]
        
        if not job_rw:
            return current_time
        
        job_works = list(job_rw.values())
        # Critical path: max job remaining work
        job_cp = max(job_works)
        
        # Machine load: total work / 5 machines (open shop, 5 machines)
        n_machines = 5
        machine_lb = total_work / n_machines
        
        # Makespan lower bound: max of job critical paths and machine load
        # Use soft max for smoothness
        lb = soft_max([job_cp, machine_lb], beta=15.0)
        return current_time + lb
    
    # --- LPT-inspired greedy with lookahead ---
    # Primary strategy: pick op with largest min_duration (LPT)
    # Tie-breaker: prefer ops with more remaining work (longer jobs)
    # Secondary: consider machine flexibility
    
    # Score each operation using LPT principle
    scores = []
    for i, f in enumerate(features):
        # LPT: prioritize long min-duration
        lpt_component = f["lpt_score"]
        # Job urgency: jobs with more total remaining work should be prioritized
        urgency_component = f["remaining_work"] * 0.3
        # Constraint: ops with fewer machine options are more critical
        constraint_component = f["constraint"] * f["min_dur"] * 0.2
        # Remaining ops in job: more remaining = higher priority
        rem_component = f["n_rem_ops"] * 0.5
        
        total_score = lpt_component + urgency_component + constraint_component + rem_component
        scores.append(total_score)
    
    # Get top candidates by score
    sorted_indices = np.argsort(scores)[::-1]
    top_candidates = sorted_indices[:min(5, n)]
    
    # Lookahead: for each top candidate, simulate one step and estimate makespan
    best_score = float('inf')
    best_idx = int(top_candidates[0])
    
    for idx in top_candidates:
        if time_module.time() > deadline:
            break
        
        # Simulate: pick this op, assign to min-duration machine
        op = ready_operations[idx]
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        comp_t = time + min_dur
        
        # Build new ready list
        new_ready = []
        for j, other in enumerate(ready_operations):
            if j == idx:
                rem = op.get("remaining_ops", [])
                if isinstance(rem, list) and len(rem) > 0:
                    next_op = rem[0].copy()
                    next_op["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                    if isinstance(rem, list) and len(rem) > 1:
                        next_op["remaining_ops"] = rem[1:]
                    else:
                        next_op["remaining_ops"] = []
                    new_ready.append(next_op)
            else:
                new_ready.append(other)
        
        # Estimate makespan from new state
        est = estimate_makespan(new_ready, comp_t)
        
        if est < best_score:
            best_score = est
            best_idx = int(idx)
    
    return best_idx
