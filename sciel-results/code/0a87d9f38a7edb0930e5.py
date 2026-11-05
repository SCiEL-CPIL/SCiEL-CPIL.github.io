"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track start time for deadline guard
_START_TIME = _time.time()
_DEADLINE = 120.0  # seconds, with safety margin below

def _estimate_machine_load(ready_operations, machine):
    """
    Estimate the future load on a given machine by summing durations of
    all ready operations that could potentially use this machine.
    This is a proxy for how congested the machine will be.
    """
    total = 0.0
    for op in ready_operations:
        opts = op["machine_options"]
        if machine in opts:
            total += opts[machine]
    return total

def dispatch(ready_operations, time):
    # Deadline guard: if we're close to the limit, fall back to a simple rule
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Collect unique machines across all ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    # Pre-compute machine load estimates (future congestion proxy)
    machine_loads = {}
    for m in all_machines:
        machine_loads[m] = _estimate_machine_load(ready_operations, m)
    
    # For each ready operation, compute regret = best_cost - second_best_cost
    # where cost = duration + alpha * estimated_machine_load
    # High regret means the machine choice matters a lot -> schedule it first
    alpha = 0.5  # weight for future load term
    
    regrets = np.zeros(n_ops)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Compute cost for each eligible machine
        costs = []
        for m, dur in opts.items():
            load = machine_loads.get(m, 0.0)
            cost = dur + alpha * load
            costs.append(cost)
        
        if len(costs) <= 1:
            # Only one machine option: no regret, but use negative cost as tiebreaker
            # (prefer shorter duration / less loaded machine implicitly)
            regrets[i] = 0.0
        else:
            # Sort costs, regret = second_best - best (positive means high impact)
            costs_sorted = sorted(costs)
            regrets[i] = costs_sorted[1] - costs_sorted[0]
    
    # Combine regret with a small urgency term (remaining work) to break ties
    # and ensure we don't ignore jobs with lots of remaining work entirely
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # Normalize remaining work to [0, 1] scale for blending
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    norm_rw = remaining_works / max_rw
    
    # Final score: primarily regret, secondarily urgency
    # This ensures we pick operations where machine choice is most critical,
    # but among equal-regret ops, we prefer those with more remaining work
    scores = regrets + 0.1 * norm_rw
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
