"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Machine-load balancing - prioritize operations targeting the most
    constrained machine (fewest options or highest load proxy) with shortest
    durations, preventing single-machine bottlenecks that dominate makespan.
    
    Returns the index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features for each operation
    remaining_works = []
    min_durations = []
    max_durations = []
    num_machines = []
    machine_options_list = []
    
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        machines = list(op["machine_options"].keys())
        if not durations:
            remaining_works.append(0)
            min_durations.append(float('inf'))
            max_durations.append(0)
            num_machines.append(1)
            machine_options_list.append([])
        else:
            remaining_works.append(work)
            min_durations.append(min(durations))
            max_durations.append(max(durations))
            num_machines.append(len(durations))
            machine_options_list.append(list(zip(machines, durations)))
    
    remaining_works = np.array(remaining_works, dtype=float)
    min_durations = np.array(min_durations, dtype=float)
    max_durations = np.array(max_durations, dtype=float)
    num_machines = np.array(num_machines, dtype=float)
    
    # Handle edge cases
    if np.all(remaining_works == 0):
        return int(np.argmin(min_durations))
    
    # Machine-load balancing strategy:
    # 1. Identify the most constrained machine (fewest available options across all ops)
    # 2. Prioritize operations that target this machine with shortest durations
    
    # Count how many operations can use each machine
    machine_usage = {}
    for opts in machine_options_list:
        for machine, dur in opts:
            if machine not in machine_usage:
                machine_usage[machine] = 0
            machine_usage[machine] += 1
    
    # Most constrained machine = one with fewest operations that can use it
    # (fewer options means less flexibility, so it's more critical)
    if machine_usage:
        most_constrained_machine = min(machine_usage.keys(), key=lambda m: machine_usage[m])
        constraint_score = machine_usage[most_constrained_machine]
    else:
        most_constrained_machine = None
        constraint_score = 0
    
    # Compute machine-specific scores
    machine_scores = np.zeros(n)
    for i in range(n):
        opts = machine_options_list[i]
        if not opts:
            machine_scores[i] = 0
            continue
        
        # Check if this operation can use the most constrained machine
        if most_constrained_machine is not None:
            for machine, dur in opts:
                if machine == most_constrained_machine:
                    # Operation targets the bottleneck machine - prioritize it
                    # Shorter duration on bottleneck machine is better
                    machine_scores[i] = 1.0 - (dur / max_durations[i]) if max_durations[i] > 0 else 1.0
                    break
        
        # Also consider: operations with fewer machine options are less flexible
        # and should be prioritized to avoid blocking
        flexibility_penalty = 1.0 / (num_machines[i] + 1)
        machine_scores[i] += 0.3 * flexibility_penalty
    
    # Normalize machine scores
    if np.max(machine_scores) > np.min(machine_scores):
        machine_norm = (machine_scores - np.min(machine_scores)) / (np.max(machine_scores) - np.min(machine_scores))
    else:
        machine_norm = np.zeros(n)
    
    # Normalize other features
    work_range = remaining_works.max() - remaining_works.min()
    if work_range > 0:
        work_norm = (remaining_works - remaining_works.min()) / work_range
    else:
        work_norm = np.zeros(n)
    
    min_dur_range = min_durations.max() - min_durations.min()
    if min_dur_range > 0:
        min_dur_norm = (min_durations - min_durations.min()) / min_dur_range
    else:
        min_dur_norm = np.zeros(n)
    
    # Composite score: machine-load balancing primary, remaining work secondary
    # Higher score = better
    w_machine = 0.6
    w_work = 0.3
    w_min_dur = 0.1
    
    # For min_duration, we want shorter to be better, so subtract
    score = (w_machine * machine_norm + w_work * work_norm - w_min_dur * min_dur_norm)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    
    return best_idx
# EVOLVE-BLOCK-END
