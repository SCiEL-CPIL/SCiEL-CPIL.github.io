"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Dynamic machine load tracking for bottleneck-aware dispatch
_machine_loads = {}
_last_dispatch_time = -1.0


def dispatch(ready_operations, time):
    global _machine_loads, _last_dispatch_time

    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the least-loaded eligible machine
            best_machine = min(machine_options, key=lambda m: _machine_loads.get(m, 0.0))
            _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + machine_options[best_machine]
            _last_dispatch_time = time
            return (0, best_machine)
        return 0

    # Update machine loads based on elapsed simulation time
    # (approximate: assume machines work continuously between dispatch calls)
    for m in list(_machine_loads.keys()):
        if _machine_loads[m] <= time:
            del _machine_loads[m]

    # For each ready operation, compute a bottleneck score:
    # score = estimated completion time on least-loaded eligible machine
    # Lower is better. We pick the operation with the highest "urgency-adjusted" priority.
    #
    # Structural change: instead of static job-centric features, we use
    # dynamic machine-centric look-ahead: which op, if started now on its
    # least-loaded machine, would have the tightest slack relative to job urgency?
    
    best_idx = 0
    best_score = -np.inf
    best_machine_choice = None

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Find the least-loaded eligible machine and its estimated completion time
        best_m_load = np.inf
        best_m_dur = np.inf
        best_m_id = None
        for m, dur in mo.items():
            m_load = _machine_loads.get(m, 0.0)
            # Estimated completion if started now: max(time, m_load) + dur
            est_completion = max(time, m_load) + dur
            # We prefer machines with lower estimated completion
            if est_completion < best_m_load:
                best_m_load = est_completion
                best_m_dur = dur
                best_m_id = m
        
        if best_m_id is None:
            continue
        
        # Bottleneck score: prioritize operations that are:
        # 1. High remaining work (critical path length)
        # 2. Can complete quickly on a currently available machine
        # 3. Have few remaining ops (closer to job completion)
        
        # Look-ahead: if we start this op now, when does it finish?
        est_finish = best_m_load
        
        # Urgency: more remaining work = more critical
        # Efficiency: shorter duration on available machine = better
        # Slack: est_finish relative to job's total remaining work
        
        # Score: higher = more urgent to dispatch now
        # Use ratio: remaining_work / est_finish (work-to-time ratio)
        # This captures "how much job progress per unit of machine time"
        
        score = remaining_work / (est_finish + 1e-9)
        
        # Small bonus for operations close to completing their job
        if remaining_ops <= 2:
            score *= 1.1
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine_choice = best_m_id

    # Update machine load for the chosen operation
    if best_machine_choice is not None:
        mo = ready_operations[best_idx]["machine_options"]
        if best_machine_choice in mo:
            _machine_loads[best_machine_choice] = max(time, _machine_loads.get(best_machine_choice, 0.0)) + mo[best_machine_choice]
    
    _last_dispatch_time = time
    
    if best_machine_choice is not None:
        return (best_idx, best_machine_choice)
    return best_idx
# EVOLVE-BLOCK-END
