"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _soft_max(values, temperature=1.0):
    """Smooth surrogate for max using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    vals = np.array(values, dtype=np.float64)
    vmax = np.max(vals)
    if temperature < 1e-10:
        return float(vmax)
    exp_vals = np.exp((vals - vmax) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return float(vmax + temperature * log_sum)


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-compute for each operation
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    job_ids = []
    machine_durations = []  # list of dicts: machine -> duration
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        job_ids.append(op["job"])
        machine_durations.append(opts)

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Graph-theoretic min-max degree reduction strategy:
    # Model as bipartite graph: jobs and machines are nodes, ready operations are edges.
    # Job degree = remaining_ops (number of operations still to process)
    # Machine degree = total remaining work on that machine (proxy for load)
    # 
    # When we remove edge i (start operation i on machine m):
    #   - Job j's degree decreases by 1 (one fewer remaining op)
    #   - Machine m's degree decreases by duration[i][m]
    # 
    # We want to pick the edge whose removal minimizes the maximum remaining degree.
    # 
    # To estimate machine loads, we sum up durations from all ready operations.
    # This is an approximation since non-ready operations also contribute, but it
    # captures the relative load distribution among machines.

    # Estimate current machine loads from ready operations
    machine_loads = defaultdict(float)
    for i in range(n):
        opts = machine_durations[i]
        for m, dur in opts.items():
            machine_loads[m] += dur

    # Convert to dict for easy access
    machine_loads_dict = dict(machine_loads)

    # For each candidate operation, compute the resulting max degree after removal
    # We use a soft-max (log-sum-exp) as a smooth surrogate for the max
    temperature = 0.5  # small temperature for near-exact max behavior
    
    scores = []
    for i in range(n):
        opts = machine_durations[i]
        job = job_ids[i]
        
        # Compute resulting job degrees after removing this operation
        # Job i's job loses one operation
        # Other jobs' degrees remain unchanged (they don't share this edge)
        
        # The key insight: in open-shop, each job has exactly one operation per machine.
        # So removing operation i reduces job j_i's remaining ops by 1.
        
        # For machine loads:
        # Machine m loses duration[i][m] for each machine m in opts
        # But wait - an operation only goes to ONE machine. So only the chosen machine's load changes.
        
        # For each possible machine assignment, compute the resulting max degree
        # and pick the machine that gives the best (lowest) max degree.
        
        best_machine_score = float('inf')
        best_machine_for_op = None
        
        for m, dur in opts.items():
            # After assigning operation i to machine m:
            # - Job j_i's remaining ops decreases by 1
            # - Machine m's load decreases by dur
            
            # New machine loads
            new_machine_loads = {}
            for mm, load in machine_loads_dict.items():
                if mm == m:
                    new_machine_loads[mm] = max(0.0, load - dur)
                else:
                    new_machine_loads[mm] = load
            
            # New job degrees: only job j_i changes
            # We need to estimate all job degrees. Since we only have ready operations,
            # we approximate job degree as remaining_ops for that job.
            # For jobs not in ready_operations, their degree is unknown but fixed.
            
            # The max degree is over all jobs and all machines.
            # We can only compute the change for job j_i and machine m.
            
            # Approximate: compute max over ready operations' jobs and all machines
            job_degrees = {}
            for k in range(n):
                jk = job_ids[k]
                if jk == job:
                    job_degrees[jk] = rem_ops[k] - 1  # this job loses one op
                else:
                    job_degrees[jk] = rem_ops[k]
            
            # Combine all degrees (jobs + machines)
            all_degrees = []
            for j, deg in job_degrees.items():
                all_degrees.append(float(deg))
            for mm, load in new_machine_loads.items():
                all_degrees.append(load)
            
            # Use soft-max as smooth surrogate for max
            sm = _soft_max(all_degrees, temperature)
            
            if sm < best_machine_score:
                best_machine_score = sm
                best_machine_for_op = m
        
        scores.append((best_machine_score, best_machine_for_op))

    # Pick the operation with the lowest resulting max degree
    # Tie-break: prefer smaller min_dur, then larger remaining_work
    order = sorted(
        range(n),
        key=lambda i: (
            scores[i][0],
            min_durs[i],
            -rem_works[i],
            rem_ops[i],
        ),
    )
    
    idx = order[0]
    chosen_machine = scores[idx][1]
    
    # Fallback if no machine was selected (shouldn't happen)
    if chosen_machine is None:
        chosen_machine = best_machines[idx]
    
    return (idx, chosen_machine)
# EVOLVE-BLOCK-END
