"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Multi-factor heuristic combining:
    1. Remaining work (total processing time left for the job) - prioritize jobs with more work
    2. Machine flexibility - prefer operations that can run on more machines (less blocking)
    3. Minimum duration - shorter operations can finish sooner, freeing up machines
    4. Remaining ops count - jobs with fewer remaining ops are closer to completion
    5. Average duration - shorter average processing time
    
    Returns: index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features for each ready operation
    remaining_works = []
    remaining_ops_counts = []
    min_durations = []
    machine_counts = []
    avg_durations = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        durations = list(mo.values())
        min_dur = min(durations) if durations else 0
        avg_dur = np.mean(durations) if durations else 0
        num_machines = len(mo.keys())
        
        remaining_works.append(rw)
        remaining_ops_counts.append(ro)
        min_durations.append(min_dur)
        machine_counts.append(num_machines)
        avg_durations.append(avg_dur)
    
    remaining_works = np.array(remaining_works, dtype=float)
    remaining_ops_counts = np.array(remaining_ops_counts, dtype=float)
    min_durations = np.array(min_durations, dtype=float)
    machine_counts = np.array(machine_counts, dtype=float)
    avg_durations = np.array(avg_durations, dtype=float)
    
    # Normalize each feature to [0, 1] range for consistent weighting
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-9:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_rw = normalize(remaining_works)
    norm_ro = normalize(remaining_ops_counts)
    norm_min_dur = normalize(min_durations)
    norm_mach_count = normalize(machine_counts)
    norm_avg_dur = normalize(avg_durations)
    
    # Weighted score:
    # - High remaining work: + (prioritize jobs with more total work left)
    # - Low remaining ops: + (jobs closer to completion should be finished)
    # - Low min duration: + (shorter ops finish faster, reduce makespan)
    # - High machine count: + (more flexible, less likely to block)
    # - Low avg duration: + (shorter average processing time)
    
    w_rw = 0.35
    w_ro = 0.20
    w_min_dur = 0.25
    w_mach = 0.10
    w_avg_dur = 0.10
    
    score = (
        w_rw * norm_rw +
        w_ro * (1 - norm_ro) +  # Invert: fewer remaining ops is better
        w_min_dur * (1 - norm_min_dur) +  # Invert: shorter min duration is better
        w_mach * norm_mach_count +
        w_avg_dur * (1 - norm_avg_dur)  # Invert: shorter avg duration is better
    )
    
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
