"""Open-shop scheduling dispatch rule using regret-based priority."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

# Deadline guard: evaluator kills after 150s wall-clock; use a safety margin.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 140.0  # seconds, leave 10s margin


def _elapsed():
    return _time.monotonic() - _DEADLINE_START


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Regret-based priority. For each ready operation, compute the "regret"
    of NOT scheduling it now: the opportunity cost in terms of future congestion.
    We estimate this by considering, for each operation, the least-loaded machine it
    can run on, and the delay that machine would face if this operation is not started.
    
    The rule selects the operation with the highest regret (most urgent to schedule).
    
    Returns: index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline check
    if _elapsed() > _DEADLINE_LIMIT - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Gather machine options and remaining work
    # For each op, find the minimum duration across its machine options
    min_durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(0)
    min_durations = np.array(min_durations, dtype=float)

    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=int)

    # Estimate machine load: for each machine, sum of durations of ops that can use it
    # and haven't been scheduled yet (all ready ops are unscheduled)
    # We approximate machine load by looking at all ready ops' options
    machine_load = defaultdict(float)
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] += dur

    # For each operation, compute regret:
    # Regret = how much worse the schedule becomes if we delay this op
    # Approximation: 
    #   1. The operation's own remaining work (larger = more urgent)
    #   2. The congestion on its best machine (if we don't schedule it, that machine's queue grows)
    #   3. The number of remaining ops in its job (more remaining = higher priority to keep pipeline moving)
    
    # Compute per-operation regret score
    regrets = np.zeros(n, dtype=float)
    
    for i in range(n):
        op = ready_operations[i]
        mo = op["machine_options"]
        
        if not mo:
            regrets[i] = remaining_work[i]
            continue
        
        # Find the least-loaded machine this op can run on
        best_machine = None
        best_load = float('inf')
        for m, dur in mo.items():
            load = machine_load.get(m, 0.0)
            if load < best_load:
                best_load = load
                best_machine = m
        
        # Regret components:
        # - remaining_work: more work = more urgent
        # - congestion: how loaded the best machine is (relative to its total)
        # - remaining_ops: more ops left in job = higher priority
        # - min_duration: shorter ops on congested machines are more urgent (blocking)
        
        # Normalize congestion: best_load / (sum of all machine loads + eps)
        total_load = sum(machine_load.values()) + 1e-9
        congestion = best_load / total_load
        
        # Weighted regret
        # Scale factors to balance terms
        rw_norm = remaining_work[i] / (np.max(remaining_work) + 1e-9)
        ro_norm = remaining_ops[i] / (np.max(remaining_ops) + 1e-9) if np.max(remaining_ops) > 0 else 0
        dur_norm = min_durations[i] / (np.max(min_durations) + 1e-9)
        
        # Regret: prioritize ops with high remaining work, on congested machines, with many remaining ops
        # Short ops on congested machines get a boost (they're blocking)
        regret = (
            0.4 * rw_norm +
            0.3 * congestion +
            0.2 * ro_norm +
            0.1 * (1.0 - dur_norm)  # shorter ops get slight priority
        )
        
        regrets[i] = regret

    # Add a small tie-breaker based on remaining work to ensure determinism
    regrets += remaining_work * 1e-6

    return int(np.argmax(regrets))
# EVOLVE-BLOCK-END
