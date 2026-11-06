"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq
import random

# Strategy: Joint Optimization of Operation Priority and Machine Assignment.
# The current seed program selects an operation index but does not influence machine choice.
# By returning a tuple (index, machine), we can explicitly assign the operation to the 
# least contested machine among its options, reducing future bottlenecks.

def dispatch(ready_operations, time):
    """
    Dispatch rule using Joint Op-Machine Optimization.
    
    1. For each ready operation, calculate a base priority (remaining_work).
    2. For each possible machine assignment for that operation, calculate a contention penalty.
    3. Select the (operation, machine) pair that maximizes the adjusted priority.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        # If only one op, pick its best machine (least contested, though contention is 0)
        # or just the first one.
        op = ready_operations[0]
        mo = op["machine_options"]
        # Pick the machine with shortest duration? Or first? 
        # Usually shortest duration is best for clearing the op quickly.
        best_m = min(mo.keys(), key=lambda m: mo[m])
        return (0, best_m)

    # Pre-compute contention for each machine
    # contention[m] = number of ready operations that can use machine m
    machine_contention = {}
    op_data = []
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        remaining_work = op["remaining_work"]
        # Store data for this op
        op_data.append({
            "index": i,
            "remaining_work": remaining_work,
            "machine_options": mo
        })
        
        # Update contention counts
        for m in mo.keys():
            if m not in machine_contention:
                machine_contention[m] = 0
            machine_contention[m] += 1

    best_score = -1
    best_pair = (0, 0) # Fallback
    
    for data in op_data:
        idx = data["index"]
        rw = data["remaining_work"]
        mo = data["machine_options"]
        
        # For this operation, try each available machine
        for m, dur in mo.items():
            # Contention of this machine
            cont = machine_contention[m]
            
            # Heuristic: 
            # We want high remaining work (clears more backlog).
            # We want low contention (less future blocking).
            # We want short duration (frees machine sooner)? 
            # Let's combine: Score = remaining_work / (1 + contention * duration_factor)
            # If a machine is heavily contended AND the op is long, it's bad.
            # If a machine is lightly contended, it's good.
            
            # Simple ratio: Work / (1 + Contention)
            # But we also care about the specific machine's load relative to others.
            # Let's use: Score = remaining_work / (1 + cont)
            # Tie breaker: Shorter duration is better (frees machine earlier).
            # Let's refine: Score = (remaining_work + 1) / (1 + cont + 0.1 * dur)
            
            # To make it more robust, let's normalize duration.
            # But simple division is fast.
            score = rw / (1.0 + cont)
            
            # If scores are very close, prefer shorter duration to free machine
            # Add a small epsilon based on duration to break ties in favor of shorter ops
            # Note: dur is in the denominator of a secondary effect?
            # Let's just use score = rw / (1 + cont) and break ties by min dur.
            
            if score > best_score:
                best_score = score
                best_pair = (idx, m)
            elif abs(score - best_score) < 1e-9:
                # Tie breaker: Choose the one with shorter duration to free machine faster
                current_dur = mo[best_pair[1]]
                if dur < current_dur:
                    best_pair = (idx, m)

    return best_pair

# EVOLVE-BLOCK-END
