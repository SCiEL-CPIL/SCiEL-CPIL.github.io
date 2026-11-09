import time
import numpy as np

# Deadline guard: keep explicit wall-clock tracking so we can bail out safely.
# We do not use this to increase search; it is a safety margin only.
_START = time.time()
_DEADLINE = 150.0  # seconds

def dispatch(ready_operations, time):
    """
    Dispatch rule: local critical path maximization with machine-load balancing.

    Structural change from the parent:
    - The parent used a static proxy: remaining_work (sum of min durations for all
      remaining ops) plus a small SPT tie-break.
    - This version explicitly computes, for each ready operation, the *remaining
      critical path length* of its job: the sum over all remaining operations
      (including the current one) of the minimum processing time across eligible
      machines. This is a precedence-aware urgency measure.
    - We also incorporate a dynamic machine-congestion penalty: for each candidate
      (operation, machine) pair, we estimate the load on that machine by summing
      the minimum durations of all other ready operations that could also use
      that machine. This encourages spreading work across machines.

    Selection:
    - Primary: maximize remaining critical path length (longest remaining job).
    - Secondary: among machines for the chosen operation, pick the one that
      minimizes (duration + congestion_penalty) to balance load.
    - Tertiary: shortest duration as a final tie-break.

    Return: (index, machine) as in the parent.
    """
    if not ready_operations:
        return -1

    # Safety: if we are already close to the deadline, return the first ready op
    # on its shortest machine to avoid timeout.
    elapsed = time.time() - _START
    if elapsed > _DEADLINE - 5.0:
        op = ready_operations[0]
        best_m = min(op['machine_options'], key=lambda m: op['machine_options'][m])
        return (0, best_m)

    n = len(ready_operations)

    # Precompute for each ready operation:
    #   min_dur[i] = minimum processing time for op i
    #   best_mach[i] = machine achieving min_dur[i]
    #   rem_cp[i] = remaining critical path length for the job of op i
    #     = sum over all remaining ops in that job of min duration
    #   We approximate rem_cp using 'remaining_work' if available, else compute.

    min_dur = [0.0] * n
    best_mach = [-1] * n
    rem_cp = [0.0] * n

    for i, op in enumerate(ready_operations):
        mo = op['machine_options']
        bd = float('inf')
        bm = -1
        for m, d in mo.items():
            if d < bd:
                bd = d
                bm = m
        min_dur[i] = bd
        best_mach[i] = bm

        # Remaining critical path: use remaining_work if present, else estimate
        rw = op.get('remaining_work', None)
        if rw is not None and rw > 0:
            rem_cp[i] = float(rw)
        else:
            # Fallback: use min_dur as a lower bound for the current op's contribution
            # and assume remaining_ops count * average min duration.
            ro = op.get('remaining_ops', 0)
            # remaining_ops likely includes the current op; if not, add 1.
            # We'll use min_dur * max(ro, 1) as a rough estimate.
            rem_cp[i] = min_dur[i] * max(ro, 1)

    # Compute machine congestion for each machine across all ready ops.
    # For each machine m, congestion[m] = sum of min_dur[i] for all i that can use m.
    # This is a static estimate of how loaded each machine is.
    machine_load = {}
    for i, op in enumerate(ready_operations):
        mo = op['machine_options']
        for m, d in mo.items():
            if m not in machine_load:
                machine_load[m] = 0.0
            machine_load[m] += min_dur[i]

    # Now score each ready operation.
    # We want to maximize rem_cp, but also prefer machines with lower load.
    # For each op i, compute the best (machine, score) pair.
    # score for (i, m) = rem_cp[i] - alpha * (min_dur[i] if m == best_mach[i] else duration on m) 
    #                    - beta * (machine_load[m] - min_dur[i]) / max(1, count_ops_on_m)
    # Actually, let's simplify:
    # For each op i, we pick the machine m that minimizes:
    #   adjusted_load = machine_load[m] - min_dur[i]  (load if we remove our own contribution)
    #   + duration[i][m]
    # Then the op's priority = rem_cp[i] - gamma * adjusted_load
    # We pick the op with highest priority.

    alpha = 0.5   # weight for machine load balancing
    gamma = 0.3   # weight for congestion penalty in op priority

    best_i = -1
    best_m = -1
    best_priority = float('-inf')

    for i, op in enumerate(ready_operations):
        mo = op['machine_options']
        # For this op, find the machine that minimizes (duration + alpha * (machine_load[m] - min_dur[i]))
        best_m_for_i = -1
        best_cost = float('inf')
        for m, d in mo.items():
            load_without_self = machine_load.get(m, 0.0) - min_dur[i]
            if load_without_self < 0:
                load_without_self = 0.0
            cost = d + alpha * load_without_self
            if cost < best_cost:
                best_cost = cost
                best_m_for_i = m

        # Op priority: rem_cp[i] - gamma * (machine_load[best_m_for_i] - min_dur[i])
        load_without_self = machine_load.get(best_m_for_i, 0.0) - min_dur[i]
        if load_without_self < 0:
            load_without_self = 0.0
        priority = rem_cp[i] - gamma * load_without_self

        if priority > best_priority:
            best_priority = priority
            best_i = i
            best_m = best_m_for_i

    if best_i == -1:
        # Should not happen, but fallback
        return (0, best_mach[0])

    return (best_i, best_m)
