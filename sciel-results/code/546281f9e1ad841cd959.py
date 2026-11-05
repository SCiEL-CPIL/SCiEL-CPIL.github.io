"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global state to track machine load across dispatch calls
# Keys are machine IDs, values are estimated busy-until times
_machine_load = {}
_call_start_time = time.time()
_DEADLINE = 1.0  # 1 second safety margin per dispatch call

def dispatch(ready_operations, time):
    global _machine_load, _call_start_time, _DEADLINE
    
    # Deadline guard: if we've been running too long, fallback to simple heuristic
    if time.time() - _call_start_time > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    if not ready_operations:
        return 0
        
    n = len(ready_operations)
    
    # 1. Estimate the completion time for each ready operation on its best machine
    # We look at machine_options. For each option, we estimate when that machine 
    # would be free (based on _machine_load) plus the duration.
    # We assume the dispatch rule will pick the machine that results in the earliest start.
    
    best_completion_times = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        min_completion = float('inf')
        best_machine = None
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback if no machine options
            best_completion_times.append(float('inf'))
            best_machines.append(None)
            continue
            
        for machine, duration in machine_options.items():
            # Get current estimated load for this machine
            # _machine_load stores the time until which the machine is busy
            current_load = _machine_load.get(machine, 0)
            
            # The operation can start at max(current_time, current_load)
            # Since all ops are "ready", they can start immediately if machine is free.
            # The completion time would be max(time, current_load) + duration
            start_time = max(time, current_load)
            completion_time = start_time + duration
            
            if completion_time < min_completion:
                min_completion = completion_time
                best_machine = machine
        
        best_completion_times.append(min_completion)
        best_machines.append(best_machine)
        
    # 2. Score each operation
    # Heuristic: Prioritize operations that finish soonest on their best machine,
    # but also consider the remaining work to avoid starving long jobs.
    # A common effective rule is "Shortest Processing Time" (SPT) on the machine,
    # or "Critical Path" (remaining work).
    # Let's try a weighted combination:
    # Score = w1 * (normalized completion time) + w2 * (normalized remaining work)
    # Lower score is better.
    
    # Normalize completion times
    min_comp = min(best_completion_times) if best_completion_times else 0
    max_comp = max(best_completion_times) if best_completion_times else 0
    range_comp = max_comp - min_comp if max_comp > min_comp else 1
    
    # Normalize remaining work
    rem_works = [op["remaining_work"] for op in ready_operations]
    min_rw = min(rem_works) if rem_works else 0
    max_rw = max(rem_works) if rem_works else 0
    range_rw = max_rw - min_rw if max_rw > min_rw else 1
    
    scores = []
    for i in range(n):
        norm_comp = (best_completion_times[i] - min_comp) / range_comp
        norm_rw = (rem_works[i] - min_rw) / range_rw
        
        # Weighting: 0.7 for congestion (completion time), 0.3 for remaining work
        # This balances short-term machine efficiency with long-term job completion
        score = 0.7 * norm_comp + 0.3 * norm_rw
        scores.append(score)
        
    # Pick the operation with the lowest score (best)
    best_idx = int(np.argmin(scores))
    
    # Update _machine_load for the chosen machine
    chosen_op = ready_operations[best_idx]
    chosen_machine = best_machines[best_idx]
    
    if chosen_machine is not None:
        # Update the load for the chosen machine
        # The machine will be busy until best_completion_times[best_idx]
        # Note: This is a simplification. In a real simulation, the machine might
        # have other jobs queued. But for a greedy heuristic, assuming the
        # machine is free after this job is a reasonable approximation for
        # immediate future dispatch decisions.
        
        # Actually, a better update is:
        # The machine is now busy until completion_time.
        # If there were other jobs, this would be wrong, but we don't have that info.
        # We assume the machine is dedicated to this job for the duration.
        
        # We update _machine_load[machine] to be the completion time.
        # This tells future dispatch calls that this machine is busy until then.
        
        # However, if multiple ops are dispatched in quick succession (same time step),
        # this might be inaccurate. But typically, dispatch is called when a machine
        # becomes free or a new job becomes ready.
        
        # Let's refine: 
        # If we assume the simulation calls dispatch for each machine becoming free,
        # then _machine_load should track when the machine is next available.
        
        # For now, set it to the completion time.
        _machine_load[chosen_machine] = best_completion_times[best_idx]
        
    return best_idx

# EVOLVE-BLOCK-END
