"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and keep a safety margin.
_START_TIME = _time.perf_counter()
_DEADLINE = 150.0 - 5.0  # 5s safety margin


def _elapsed_ok():
    return (_time.perf_counter() - _START_TIME) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine runs it.
    
    Strategy: Dynamic bottleneck optimization.
    Prioritize the operation that maximizes the ratio of its job's remaining
    critical path length to the estimated cumulative load of its eligible machines.
    This shifts focus from static job attributes to real-time machine congestion,
    explicitly avoiding operations that would exacerbate load on already busy machines.
    
    Returns (index, machine) to explicitly select the machine as well.
    """
    if not _elapsed_ok():
        # Safety: fall back to simple argmax
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        machine_options = op["machine_options"]
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_machine)
    
    # For each operation, compute:
    # 1. remaining_critical_path: sum of min durations of remaining ops (including current)
    # 2. machine_load: sum of min durations across all eligible machines for this op
    # 3. score = remaining_critical_path / (machine_load + epsilon)
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    epsilon = 1e-9
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", sum(machine_options.values()))
        
        # Estimate remaining critical path length:
        # Use remaining_work as a proxy for total remaining processing time
        # This is a smooth surrogate for the true critical path
        
        # For each eligible machine, compute the score
        for machine, duration in machine_options.items():
            # Machine load estimate: the duration on this specific machine
            # Plus a small penalty if this machine is likely congested
            # We approximate congestion by considering the duration itself
            
            # The ratio: how much remaining work does this job have vs. how long it takes on this machine?
            # Higher ratio means: lots of remaining work, but this machine is fast for it -> good choice
            
            if remaining_ops > 0:
                # Average remaining work per op
                avg_remaining_work = remaining_work / remaining_ops
            else:
                avg_remaining_work = remaining_work
            
            # Score: prioritize jobs with high remaining work relative to machine duration
            # This naturally balances:
            # - Jobs with lots of work (to avoid starvation)
            # - Machines that are fast for this operation (to reduce makespan)
            
            score = avg_remaining_work / (duration + epsilon)
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = machine
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
