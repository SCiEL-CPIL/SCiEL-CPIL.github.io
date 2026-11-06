"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard: 150s wall-clock limit, safety margin 5s
_DEADLINE = _time.time() + 145.0


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: A composite heuristic that considers:
    1. Critical path length (remaining work) - prioritize jobs with more work left
    2. Shortest processing time on the best machine option
    3. Number of remaining operations (jobs closer to completion get priority)
    4. Machine load balancing via soft-min of available machine durations
    
    Uses a smooth surrogate (soft-min) for machine selection to avoid ties
    and provide a continuous signal for machine preference.
    """
    # Deadline check
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute machine load estimates from all ready operations
    # This helps with load balancing
    machine_loads = {}
    for op in ready_operations:
        mo = op["machine_options"]
        for machine, duration in mo.items():
            if machine not in machine_loads:
                machine_loads[machine] = 0.0
            machine_loads[machine] += duration

    scores = []
    for i, op in enumerate(ready_operations):
        # Find the minimum duration across all machine options
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
            # Use soft-min for smoother machine preference signal
            # Soft-min gives a value slightly above the true min, 
            # with larger beta being sharper
            beta = 3.0
            vals = list(machine_options.values())
            min_val = min(vals)
            shifted = [v - min_val for v in vals]
            exp_sum = sum(math.exp(-beta * s) for s in shifted)
            soft_min_duration = min_val - (1.0 / beta) * math.log(exp_sum)
        else:
            min_duration = 0
            soft_min_duration = 0

        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Machine load penalty: prefer operations that go to less-loaded machines
        # Estimate which machine this op would likely use (the one with min duration)
        best_machine = None
        best_dur = float('inf')
        for m, d in machine_options.items():
            if d < best_dur:
                best_dur = d
                best_machine = m
        
        machine_load_penalty = 0.0
        if best_machine is not None and best_machine in machine_loads:
            # Normalize load penalty
            machine_load_penalty = 0.2 * machine_loads[best_machine]

        # Composite score:
        # - Higher remaining_work is better (prioritize jobs with more work - critical path)
        # - Lower min_duration is better (shortest processing time first)
        # - Lower remaining_ops is better (prioritize jobs closer to completion)
        # - Lower machine load is better (load balancing)
        
        # Weights tuned to balance critical path, SPT, and load balancing
        score = (
            1.5 * remaining_work -           # Critical path: prioritize more remaining work
            1.0 * soft_min_duration +         # SPT: prefer shorter processing (smoothed)
            0.5 * remaining_ops -             # Prefer fewer remaining ops
            machine_load_penalty              # Load balancing penalty
        )
        scores.append(score)

    # Pick the operation with the highest composite score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
