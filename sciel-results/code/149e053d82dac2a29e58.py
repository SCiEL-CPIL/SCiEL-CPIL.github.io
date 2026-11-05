"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

import math
import time as _time

# Module-level state for tracking machine load estimates across dispatch calls
# We maintain a running estimate of how much work each machine has queued
_mach_load_estimate = {}
_last_dispatch_time = 0.0
_last_total_remaining = 0.0

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy (flow-network bottleneck model):
    1. Model each ready operation as a path: job_source -> operation -> machine_sink
    2. The "capacity" of the path is the bottleneck (minimum residual time) along it
    3. Score each operation by the maximum bottleneck capacity to the least-loaded machine
    4. Prioritize operations that unlock the most downstream work (remaining_ops)
    5. Use soft-min for smooth machine selection considering all options
    6. Track machine load estimates to balance work across machines
    """
    global _mach_load_estimate, _last_dispatch_time, _last_total_remaining

    if not ready_operations:
        return 0

    # Deadline guard: if we've been running too long, return a safe default
    _start_time = _time.time()
    _DEADLINE_MARGIN = 10.0  # seconds before the 120s limit
    _MAX_TIME = 120.0 - _DEADLINE_MARGIN

    # Temperature parameter for soft-min
    temp = 1.5

    # Update machine load estimates based on time elapsed since last dispatch
    # If time has advanced, machines have been processing work
    if _last_dispatch_time > 0 and time > _last_dispatch_time:
        dt = time - _last_dispatch_time
        for m in list(_mach_load_estimate.keys()):
            _mach_load_estimate[m] = max(0.0, _mach_load_estimate[m] - dt)
    _last_dispatch_time = time

    # Compute total remaining work for normalization
    total_remaining = sum(op.get('remaining_work', 0) for op in ready_operations)
    if total_remaining <= 0:
        total_remaining = 1.0
    _last_total_remaining = total_remaining

    # Count how many ready ops each machine is a candidate for
    mach_candidate_count = {}
    for op in ready_operations:
        for m in op.get('machine_options', {}):
            mach_candidate_count[m] = mach_candidate_count.get(m, 0) + 1

    best_index = 0
    best_score = -float('inf')
    best_machine = None
    best_bottleneck = float('inf')
    best_downstream = 0

    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        remaining_ops = op.get('remaining_ops', 1)

        if not machine_options:
            continue

        # For each machine option, compute the "bottleneck capacity" of the path
        # Path: job_source -> this_op -> machine_sink
        # Capacity = min(remaining_work / remaining_ops, duration, 1/(load+1))
        # We want to find the machine that maximizes this bottleneck

        best_mach_bottleneck = 0.0
        best_mach_for_op = None
        best_mach_dur = float('inf')

        # Soft-min log-sum-exp for smooth machine selection
        log_sum = 0.0

        for machine_id, duration in machine_options.items():
            # Estimate current load on this machine
            load = _mach_load_estimate.get(machine_id, 0.0)

            # Bottleneck components:
            # 1. How much work this job still has per remaining op (urgency signal)
            work_per_op = remaining_work / max(1, remaining_ops)

            # 2. Processing time on this machine
            proc_time = duration

            # 3. Machine availability factor: less loaded machines have higher capacity
            #    Use soft inverse: 1 / (1 + load * candidate_pressure)
            #    candidate_pressure: more ops competing for this machine = lower capacity
            n_candidates = mach_candidate_count.get(machine_id, 1)
            pressure = 1.0 + 0.5 * (n_candidates - 1)
            avail_factor = 1.0 / (1.0 + load * 0.3 * pressure)

            # 4. Downstream unlock: how many more ops does completing this enable?
            #    remaining_ops includes this op, so downstream = remaining_ops - 1
            downstream = max(0, remaining_ops - 1)

            # Bottleneck = minimum of the path capacities
            # We normalize each component to be comparable
            # Scale: use the minimum of (work_per_op, proc_time * avail_factor)
            # as the bottleneck, weighted by downstream unlock potential

            # Effective capacity of this path
            path_capacity = min(work_per_op, proc_time * avail_factor)

            # Score contribution: bottleneck * downstream_unlock_bonus
            # Operations that unlock more downstream work get higher priority
            downstream_bonus = 1.0 + 0.2 * downstream

            op_mach_score = path_capacity * downstream_bonus

            # For soft-min: accumulate
            log_sum += math.exp(-duration / temp)

            if op_mach_score > best_mach_bottleneck:
                best_mach_bottleneck = op_mach_score
                best_mach_for_op = machine_id
                best_mach_dur = duration

        if best_mach_for_op is None:
            continue

        # Soft-min effective duration (smooth consideration of all machine options)
        if log_sum > 0:
            soft_min_dur = -temp * math.log(log_sum)
        else:
            soft_min_dur = best_mach_dur

        # Final score: combine bottleneck capacity with urgency and efficiency
        epsilon = 1e-6

        # Urgency: high remaining work relative to time
        urgency = remaining_work / (soft_min_dur + epsilon)

        # Efficiency: prefer shorter effective processing
        efficiency = 1.0 / (soft_min_dur + epsilon)

        # Downstream unlock: prioritize ops that free up more future work
        downstream_unlock = max(0, remaining_ops - 1)

        # Combine scores
        # Primary: bottleneck capacity (flow-network model)
        # Secondary: urgency and efficiency
        score = best_mach_bottleneck * 0.5 + urgency * 0.25 + efficiency * 0.15 + downstream_unlock * 0.1

        # Update machine load estimate for the selected machine
        # (We'll commit after selection, but estimate here for scoring)

        # Tie-breaking: prefer shorter bottleneck, then fewer remaining ops
        if score > best_score + 1e-12:
            best_score = score
            best_index = i
            best_machine = best_mach_for_op
            best_bottleneck = best_mach_dur
            best_downstream = remaining_ops
        elif abs(score - best_score) <= 1e-12:
            if best_mach_dur < best_bottleneck - 1e-9:
                best_index = i
                best_machine = best_mach_for_op
                best_bottleneck = best_mach_dur
                best_downstream = remaining_ops
            elif abs(best_mach_dur - best_bottleneck) <= 1e-9:
                if remaining_ops < best_downstream:
                    best_index = i
                    best_machine = best_mach_for_op
                    best_bottleneck = best_mach_dur
                    best_downstream = remaining_ops

    # Update machine load estimate for the selected operation's machine
    if best_machine is not None:
        selected_op = ready_operations[best_index]
        selected_dur = selected_op.get('machine_options', {}).get(best_machine, 0)
        _mach_load_estimate[best_machine] = _mach_load_estimate.get(best_machine, 0.0) + selected_dur

    # Deadline check
    if _time.time() - _start_time > _MAX_TIME:
        return (best_index, best_machine)

    return (best_index, best_machine)
