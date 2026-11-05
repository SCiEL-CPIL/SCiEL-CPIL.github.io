"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: if we've used > 140s, fall back to simple heuristic to avoid timeout
    if time.time() - _start_time > 140:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    best_score = float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Estimate job completion if we start this op now
        # Priority: high remaining_work is urgent, but we also care about machine availability
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        # Find the machine that finishes this operation earliest
        # We don't know exact machine availability, so we assume 'time' is the current global time.
        # A better proxy for machine congestion is to pick the machine with the shortest processing time
        # if we assume machines are idle or we can't see their queue. 
        # However, the task allows returning (index, machine).
        # If we return just index, the simulator might pick the machine. 
        # The prompt says: "optionally, which eligible machine runs it... dispatch(ready_operations, time) -> index ... (optionally (index, machine))"
        
        # Strategy: 
        # 1. Calculate a priority score for the operation.
        #    High remaining_work means the job is critical (long critical path).
        #    Low processing time on the best available machine reduces immediate bottleneck.
        
        # Let's estimate the "urgency" as remaining_work.
        # Let's estimate the "efficiency" as 1 / min_duration.
        
        min_duration = min(machine_options.values())
        # To avoid division by zero
        if min_duration == 0:
            min_duration = 1e-9
            
        # Heuristic: We want to prioritize jobs with high remaining work, 
        # but also prefer operations that can be processed quickly to free up resources.
        # A common FJSP heuristic is Critical Ratio or similar.
        # Let's try: Score = remaining_work / (min_duration + 1)
        # This balances urgency (remaining_work) with speed (min_duration).
        
        score = op["remaining_work"] / (min_duration + 1.0)
        
        # We want to MAXIMIZE this score (high urgency, fast processing).
        # So we look for the max score.
        
        if score > best_score:
            best_score = score
            best_idx = i
            # Select the machine with the shortest duration for this operation
            # This is a simple local optimization: use the fastest machine for this op.
            best_machine = min(machine_options, key=machine_options.get)

    # The interface allows returning (index, machine) or just index.
    # Returning (index, machine) is more specific and helps the simulator avoid suboptimal machine assignment.
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
