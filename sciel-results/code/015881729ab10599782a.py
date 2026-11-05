"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


# Deadline guard: track start time to ensure we finish within budget
_DISPATCH_START = None
_DEADLINE_MARGIN = 10.0  # seconds before the 120s limit


def _ensure_deadline():
    """Check if we're approaching the deadline; return True if we should return immediately."""
    global _DISPATCH_START
    if _DISPATCH_START is None:
        _DISPATCH_START = _time.time()
    elapsed = _time.time() - _DISPATCH_START
    return elapsed > (120.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If only one operation, return it
    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're running out of time, fall back to a simple rule
    if _ensure_deadline():
        # Simple fallback: pick operation with max remaining_work
        max_work = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_work:
                max_work = rw
                best_idx = i
        return best_idx

    n = len(ready_operations)
    scores = np.zeros(n, dtype=np.float64)

    # Pre-compute machine contention: for each machine, count how many ready ops
    # target it and sum their durations. This gives a proxy for machine load.
    machine_load = {}  # machine -> [count, total_duration]
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_load:
                machine_load[machine] = [0, 0.0]
            machine_load[machine][0] += 1
            machine_load[machine][1] += duration

    # Also compute for each machine: the minimum duration among ops targeting it
    # This helps identify which ops are "short fillers" for busy machines
    machine_min_dur = {}  # machine -> min duration among ready ops targeting it
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_min_dur or duration < machine_min_dur[machine]:
                machine_min_dur[machine] = duration

    for i, op in enumerate(ready_operations):
        # Get machine options: {machine: duration}
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        if not machine_options:
            # Fallback: use remaining_work
            scores[i] = remaining_work
            continue

        durations = list(machine_options.values())
        max_duration = max(durations)
        min_duration = min(durations)

        # Constraint ratio: how much more constrained is this op vs its best alternative
        constraint_ratio = max_duration / (min_duration + 1e-9)

        # Machine contention score: for each machine this op can run on,
        # compute a "pressure" metric. Higher pressure means the machine is
        # heavily loaded, so scheduling an op there reduces fragmentation.
        machine_pressure = 0.0
        best_machine_pressure = 0.0
        for machine, duration in machine_options.items():
            if machine in machine_load:
                count, total_dur = machine_load[machine]
                # Pressure: how busy is this machine relative to this op's duration?
                # A machine with many ops and high total duration is congested
                pressure = count * (total_dur / (count + 1e-9))
                if pressure > machine_pressure:
                    machine_pressure = pressure
                # Track the pressure of the machine where this op has shortest duration
                # (most likely to be scheduled there)
                if duration == min_duration:
                    best_machine_pressure = pressure

        # Normalize machine_pressure to be comparable across ops
        # Use a soft normalization: pressure / (pressure + 1) to bound it
        pressure_factor = machine_pressure / (machine_pressure + max_duration + 1e-9)
        best_pressure_factor = best_machine_pressure / (best_machine_pressure + min_duration + 1e-9)

        # Machine availability: how many machines does this op have options on?
        # Ops with fewer machine options are more constrained and should be prioritized
        num_machines = len(machine_options)
        # In open shop, typically all ops can run on any machine, so this is often 5
        # But if some ops have restricted options, prioritize those
        availability_penalty = 1.0 / (num_machines + 1e-9)

        # Criticality: remaining_ops indicates how many more operations this job has
        # Jobs with fewer remaining ops are closer to completion and should be finished
        # to free up resources
        if remaining_ops > 0:
            criticality = 1.0 / remaining_ops
        else:
            criticality = 1.0

        # Combine scores with tuned weights:
        # 1. remaining_work: LRPT component (high remaining work = high priority)
        # 2. constraint_ratio: ops with severe bottleneck should go first
        # 3. best_pressure_factor: ops targeting congested machines on their best option
        # 4. availability_penalty: ops with fewer machine options are more constrained
        # 5. criticality: jobs closer to completion should be finished

        w_work = 1.0
        w_constraint = 0.5
        w_pressure = 0.6
        w_availability = 0.3
        w_criticality = 0.2

        scores[i] = (
            w_work * remaining_work
            + w_constraint * constraint_ratio * remaining_work
            + w_pressure * best_pressure_factor * remaining_work
            + w_availability * availability_penalty * remaining_work
            + w_criticality * criticality * remaining_work
        )

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
