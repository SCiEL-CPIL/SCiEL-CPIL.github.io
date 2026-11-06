"""Flexible job-shop dispatch: bottleneck-elimination with load-balanced machine selection."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a bottleneck-elimination strategy: prioritize the operation whose job
    has the longest remaining critical path (estimated via remaining_ops and
    remaining_work), and assign it to the least-loaded eligible machine to
    prevent congestion.
    
    Returns:
        (int, machine): index of the ready operation and preferred machine
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        mo = ready_operations[0]["machine_options"]
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Deadline guard
    _t0 = _time.time()
    DEADLINE = 115.0  # seconds; leaves margin before 120s limit

    # Gather basic info
    machine_options_list = [op["machine_options"] for op in ready_operations]
    remaining_work_list = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops_list = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    jobs_list = [op["job"] for op in ready_operations]

    # Estimate machine load: for each machine, sum of durations of all ready ops
    # that could use it. This approximates how busy each machine will be.
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    all_machines = sorted(all_machines)

    machine_load = {m: 0.0 for m in all_machines}
    for mo in machine_options_list:
        for m, dur in mo.items():
            machine_load[m] += dur

    # For each ready operation, compute:
    # 1. Critical path estimate: remaining_ops * avg_duration (proxy for how long
    #    the job will keep blocking the makespan)
    # 2. Best machine: the one with lowest (duration + load) to balance load
    # 3. Score: prioritize operations with longest remaining critical path
    
    scores = []
    best_machines = []
    
    for i, mo in enumerate(machine_options_list):
        if len(mo) == 0:
            scores.append(0.0)
            best_machines.append(None)
            continue

        # Estimate average duration for this operation
        durations = np.array(list(mo.values()), dtype=np.float64)
        avg_dur = float(np.mean(durations))
        min_dur = float(np.min(durations))

        # Critical path estimate: how much longer this job will contribute
        # to the makespan. Use remaining_ops * avg_dur as a proxy.
        # Also consider remaining_work which is the total work left.
        critical_path = remaining_ops_list[i] * avg_dur
        
        # Urgency: operations with more remaining work are more critical
        # because they represent more total time that could block completion
        urgency = remaining_work_list[i]

        # Score: combine critical path and urgency
        # The job with the longest remaining critical path is the bottleneck
        score = critical_path + 0.5 * urgency

        # Select best machine: minimize (duration + machine_load)
        # This balances load across machines
        best_m = None
        best_cost = float('inf')
        for m, dur in mo.items():
            cost = dur + machine_load.get(m, 0.0)
            if cost < best_cost:
                best_cost = cost
                best_m = m

        scores.append(score)
        best_machines.append(best_m)

    # Check deadline
    if _time.time() - _t0 > DEADLINE:
        # Fallback: pick max remaining work
        best_idx = int(np.argmax(remaining_work_list))
        mo = machine_options_list[best_idx]
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m] + machine_load.get(m, 0.0))
            return (best_idx, best_m)
        return best_idx

    # Pick the operation with the highest score (longest critical path / most urgent)
    best_idx = int(np.argmax(scores))

    # Return (index, machine) to let the simulator know our preferred machine
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
