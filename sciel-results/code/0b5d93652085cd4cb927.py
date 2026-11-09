import time as _time
import numpy as np

# Deadline guard: ensure we return well before 150s wall-clock limit
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

# Lookahead depth: number of future dispatch steps to simulate
_LOOKAHEAD_DEPTH = 5

def _simulate_makespan(ready_ops, current_time, depth):
    """
    Simulate the next `depth` dispatch events greedily (always pick min duration)
    and return the estimated makespan increase (max completion time among simulated ops).
    This is a lightweight proxy for how bad a choice is in terms of system congestion.
    """
    # Work with a shallow copy of the state to avoid mutation
    # Since we're only looking ahead, we simulate locally
    pending = [dict(op) for op in ready_ops]
    machine_free = {}  # machine -> earliest free time
    job_remaining = {} # job -> remaining work
    
    # Initialize machine free times to current_time
    # We don't know all machines, so we'll track dynamically
    # Actually, for a true lookahead, we'd need to know which machines are busy.
    # But `ready_operations` only contains ops that are READY.
    # The key insight: if we pick op i, machine m becomes busy for duration d.
    # Other ops on machine m in the ready list are NOT ready yet (blocked), 
    # but ops on other machines are.
    
    # Simplified Lookahead Model:
    # 1. Pick an op (the one we are evaluating).
    # 2. Simulate that it starts now.
    # 3. For the next `depth-1` steps, assume we always pick the ready op 
    #    with the shortest processing time (SPT) on any machine.
    # 4. Return the max completion time of all simulated ops.
    
    # However, the `ready_operations` list changes as we simulate.
    # An op becomes ready when its job's previous op finishes.
    # This is complex to simulate perfectly without the full state.
    
    # Alternative Structural Approach: **Congestion-Aware SPT**
    # Instead of full simulation, we estimate the "cost" of assigning an op to a machine
    # by looking at the sum of durations of other ready ops on that same machine.
    # This is a static approximation of congestion.
    
    # Let's try a different structural change: **Job-Weighted SPT with Machine Load**
    # Score = (Remaining Work of Job) / (Processing Time on Machine) * (1 / (1 + Load on Machine))
    # This penalizes assigning to already-loaded machines.
    
    pass

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute machine loads from the current ready set
    # "Load" is the sum of durations of all ready ops assigned to that machine
    machine_load = {}
    for op in ready_operations:
        options = op['machine_options']
        if not options:
            continue
        # Assume the op will be assigned to the machine with min duration for load estimation
        min_dur = min(options.values())
        # Find all machines with this min duration? Or just the best one?
        # For load estimation, we should consider the machine we *would* pick.
        # Let's assume the "intended" machine is the one with min duration.
        best_m = next(m for m, d in options.items() if d == min_dur)
        machine_load[best_m] = machine_load.get(best_m, 0) + min_dur
    
    best_idx = 0
    best_score = float('-inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        min_dur = min(options.values())
        best_machine_for_op = next(m for m, d in options.items() if d == min_dur)
        
        remaining_work = op['remaining_work']
        
        # Structural Change: **Bottleneck Mitigation via Machine Load Penalty**
        # Instead of just WSPT, we penalize assigning to machines that are
        # already heavily loaded with other ready operations.
        # This attempts to spread work across machines to reduce contention.
        
        load = machine_load.get(best_machine_for_op, 0)
        
        # The higher the load, the more we penalize.
        # Use a harmonic-like penalty: 1 / (1 + load)
        # This ensures that even if remaining_work is high, we avoid bottlenecks.
        
        if remaining_work <= 0:
            wspt_part = 0
        else:
            wspt_part = remaining_work / (min_dur + 1)
        
        # Combine WSPT and Load Penalty
        # We want to maximize the score.
        # High remaining work is good (priority).
        # Low load is good (less contention).
        
        # Let's try: Score = WSPT * LoadFactor
        # Where LoadFactor = 1 / (1 + load)
        # This reduces the effective priority of ops on busy machines.
        
        load_factor = 1.0 / (1.0 + load)
        score = wspt_part * load_factor
        
        # Small tie-breaker: prefer shorter processing time
        score += 0.01 / (min_dur + 1)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
            
    return (best_idx, best_machine)
