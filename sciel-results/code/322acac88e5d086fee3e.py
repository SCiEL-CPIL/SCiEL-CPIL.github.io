"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and keep a safety margin.
_START_TIME = _time.perf_counter()
_DEADLINE = 150.0 - 5.0  # 5s safety margin


def _elapsed_ok():
    return (_time.perf_counter() - _START_TIME) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine runs it.
    
    Strategy: Combine multiple heuristics:
    1. For each operation, evaluate all machine options
    2. Prioritize operations with high remaining work that can be done quickly
    3. Balance machine loads implicitly by preferring shorter durations
    
    Returns (index, machine) to explicitly select the machine as well.
    """
    if not _elapsed_ok():
        # Safety: fall back to simple argmax
        if not ready_operations:
            return 0
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0
    
    # Pre-compute features for each ready operation
    # For each op, we consider all machine options and compute a composite score
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job = op.get("job", 0)
        op_num = op.get("op", 0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0)
        
        if not machine_options:
            continue
        
        # For each eligible machine, compute a score
        for machine, duration in machine_options.items():
            # Heuristic 1: Work-to-time ratio (prioritize jobs with more work per unit time)
            # Higher remaining work and lower duration => Higher score
            if duration > 0:
                work_ratio = remaining_work / (duration + 1e-9)
            else:
                work_ratio = remaining_work
            
            # Heuristic 2: Remaining work (higher is better for making progress)
            rw_score = remaining_work
            
            # Heuristic 3: Remaining ops (prioritize jobs with more steps to prevent starvation)
            ro_score = remaining_ops
            
            # Heuristic 4: Shortest processing time preference (lower duration is better)
            spf_bonus = -duration * 0.1  # Small penalty for long durations
            
            # Composite score: weighted combination
            # Emphasize work-to-time ratio as it captures urgency and efficiency
            score = (
                0.4 * work_ratio +          # work-to-time ratio
                0.3 * rw_score +             # remaining work
                0.2 * ro_score +             # remaining ops
                spf_bonus                     # shortest processing time
            )
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = machine
    
    # Fallback if something went wrong
    if best_machine is None:
        return 0

    # Return tuple (index, machine) to explicitly assign the machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
