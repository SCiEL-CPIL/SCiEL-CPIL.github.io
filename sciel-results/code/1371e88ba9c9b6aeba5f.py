"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate metrics for each ready operation
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])

    # Deadline guard
    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # Collect all machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    # Estimate machine loads: only count each op on its best machine
    machine_loads = {m: 0.0 for m in all_machines}
    for i, op in enumerate(ready_operations):
        best_m = best_machines[i]
        machine_loads[best_m] += min_durs[i]
    
    # Smooth max with smaller temperature for sharper max approximation
    def smooth_max(loads, temperature=0.5):
        if not loads:
            return 0.0
        arr = np.array(loads, dtype=np.float64)
        if len(arr) == 1:
            return float(arr[0])
        shifted = arr - np.max(arr)
        log_sum = np.log(np.sum(np.exp(shifted / temperature)))
        return float(np.max(arr) + log_sum * temperature)
    
    load_values = list(machine_loads.values())
    if load_values:
        current_smooth_max = smooth_max(load_values, temperature=0.5)
        current_mean_load = sum(load_values) / len(load_values)
    else:
        current_smooth_max = 0.0
        current_mean_load = 0.0
    
    # Score each candidate
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        # Find least-loaded compatible machine
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        
        # Simulate assignment
        new_load = machine_loads.get(min_load_machine, 0.0) + opts[min_load_machine]
        
        new_loads = list(machine_loads.values())
        machine_list = list(machine_loads.keys())
        m_idx = machine_list.index(min_load_machine)
        new_loads[m_idx] = new_load
        
        new_smooth_max = smooth_max(new_loads, temperature=0.5)
        load_increase = new_smooth_max - current_smooth_max
        
        # Critical path: prioritize if this op is the last remaining for its job
        # or if it's a bottleneck (longest duration among remaining ops for that job)
        critical_bonus = 0.0
        if is_last_ops[i]:
            critical_bonus = 1.0
        else:
            # Estimate if this op is critical: high remaining work relative to others
            critical_bonus = 0.5 * (remaining_works[i] / max(1.0, max(remaining_works)))
        
        score = (
            load_increase,
            -critical_bonus,  # Higher critical bonus = lower score = preferred
            min_durs[i],
            -remaining_works[i]
        )
        scores.append(score)
    
    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
