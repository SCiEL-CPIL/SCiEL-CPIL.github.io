"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _logsumexp(vals, temperature=1.0):
    """Compute log-sum-exp for smooth max approximation."""
    if not vals:
        return 0.0
    max_v = max(vals)
    if max_v == float('inf'):
        return float('inf')
    if max_v == float('-inf'):
        return float('-inf')
    sum_exp = 0.0
    for v in vals:
        sum_exp += math.exp((v - max_v) / temperature)
    return max_v + temperature * math.log(sum_exp)


def _estimate_makespan_smooth(machine_loads, remaining_ops, time, temperature=0.5):
    """
    Estimate makespan using a smooth surrogate (log-sum-exp) for the max.
    Returns a scalar score where lower is better.
    """
    if not remaining_ops:
        # No more ops; makespan is determined by current machine loads
        loads = list(machine_loads.values())
        if not loads:
            return time
        return _logsumexp(loads, temperature)
    
    # Greedily assign remaining ops to machines, tracking loads
    mach_avail = dict(machine_loads)
    sim_time = time
    
    # Sort remaining ops by min duration (SPT) for greedy assignment
    ops_sorted = sorted(remaining_ops, key=lambda op: min(op["machine_options"].values()))
    
    for op in ops_sorted:
        opts = op["machine_options"]
        # Pick the machine that minimizes the resulting completion time
        best_m = None
        best_end = float('inf')
        for m, dur in opts.items():
            end = max(sim_time, mach_avail.get(m, 0.0)) + dur
            if end < best_end:
                best_end = end
                best_m = m
        if best_m is not None:
            mach_avail[best_m] = best_end
            sim_time = best_end
    
    # Use smooth max of machine availabilities as makespan estimate
    loads = list(mach_avail.values())
    if not loads:
        return sim_time
    return _logsumexp(loads, temperature)


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate basic metrics
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []
    job_remaining_work = {}

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])
        # Track max remaining work per job for CPM proxy
        job = op["job"]
        if job not in job_remaining_work or op["remaining_work"] > job_remaining_work[job]:
            job_remaining_work[job] = op["remaining_work"]

    # Deadline guard
    if not _elapsed_ok():
        # CPM proxy: pick op from job with max remaining work, break by min dur
        idx = max(range(n), key=lambda i: (remaining_works[i], -min_durs[i]))
        return (idx, best_machines[idx])

    # Collect all machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    if not all_machines:
        return (0, best_machines[0])

    # Estimate current machine loads from ready ops using greedy water-filling
    # Weight by job remaining work to emphasize critical paths
    machine_loads = {m: 0.0 for m in all_machines}
    # Sort by (min_dur, -remaining_work) for load estimation
    op_order = sorted(range(n), key=lambda i: (min_durs[i], -remaining_works[i]))
    for i in op_order:
        opts = ready_operations[i]["machine_options"]
        # Assign to least-loaded compatible machine
        best_m_for_load = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        machine_loads[best_m_for_load] += opts[best_m_for_load]

    # For each candidate operation, estimate makespan if we pick it
    # Use smooth log-sum-exp surrogate for better gradient-like behavior
    scores = []
    temperature = 0.5  # Smaller = closer to hard max, larger = smoother
    
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        best_score_for_i = float('inf')
        best_m_for_i = best_machines[i]
        
        for m, dur in opts.items():
            # If we pick this op on machine m, it adds dur to machine m's load
            new_machine_loads = dict(machine_loads)
            new_machine_loads[m] = new_machine_loads.get(m, 0.0) + dur
            
            # Build remaining ops (all except the one we picked)
            remaining_ops = []
            for j, op_j in enumerate(ready_operations):
                if j == i:
                    continue
                remaining_ops.append({
                    "machine_options": op_j["machine_options"],
                    "remaining_work": op_j["remaining_work"],
                    "remaining_ops": op_j["remaining_ops"] - 1 if op_j["remaining_ops"] > 1 else 0
                })
            
            # Smooth makespan estimate
            ms = _estimate_makespan_smooth(new_machine_loads, remaining_ops, time, temperature)
            
            if ms < best_score_for_i:
                best_score_for_i = ms
                best_m_for_i = m
        
        scores.append((best_score_for_i, best_m_for_i))
    
    # CPM proxy ranking: 
    # Primary: estimated makespan (lower is better)
    # Secondary: prefer ops from jobs with larger remaining work (critical path)
    # Tertiary: prefer last ops (they unblock the job)
    # Quaternary: SPT (shorter processing time first)
    def score_key(i):
        ms, bm = scores[i]
        job_rw = job_remaining_work.get(ready_operations[i]["job"], 0)
        return (ms, -job_rw, 0 if is_last_ops[i] else 1, min_durs[i])
    
    idx = min(range(n), key=score_key)
    best_m = scores[idx][1]
    return (idx, best_m)
# EVOLVE-BLOCK-END
