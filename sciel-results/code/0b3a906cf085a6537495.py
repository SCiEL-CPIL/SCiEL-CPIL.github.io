"""Improved dispatch rule for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Simple deadline guard: we must finish well before 120s (evaluator kills at 150s wall-clock).
# The dispatch function is called many times; we just ensure each call is fast.
# No persistent state across calls is assumed, so we keep it stateless.

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine.
    
    Strategy: WSJF-like heuristic that balances:
      - Job criticality: remaining_work (total work left in the job) and remaining_ops
      - Machine load: prefer the machine option with the shortest duration (less congestion)
      - Urgency: jobs with more remaining ops are more critical
    
    Score for each ready operation:
      urgency = remaining_ops * remaining_work  (bigger = more critical)
      best_machine_duration = min duration among machine_options
      load_penalty = best_machine_duration
      score = urgency / (1.0 + load_penalty)
    
    We return the index of the operation with the highest score.
      Optionally return (index, machine) to also pick the machine.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with the shortest duration to minimize makespan contribution
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0
    
    best_idx = 0
    best_score = -1.0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # Fallback: no machine options, just use urgency
            score = remaining_ops * remaining_work
        else:
            # Find the machine with the shortest duration (least load/congestion)
            min_duration = min(machine_options.values())
            best_machine_for_op = min(machine_options, key=lambda m: machine_options[m])
            
            # WSJF-like: urgency / (1 + load_penalty)
            # urgency captures how much work is left in this job (criticality)
            # load_penalty captures how long this op will block a machine
            urgency = remaining_ops * remaining_work
            load_penalty = min_duration
            score = urgency / (1.0 + load_penalty)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op if machine_options else None
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
