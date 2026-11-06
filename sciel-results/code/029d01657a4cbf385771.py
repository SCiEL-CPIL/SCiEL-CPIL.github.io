"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()

    # Deadline guard: evaluator kills at 150s; use 145s as safety margin.
    if _time.time() - _START_TIME > 145:
        # Fast fallback: shortest processing time
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < best_dur:
                best_dur = d
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Precompute machine load from the ready queue for contention awareness
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0) + d

    # Also track max load for normalization
    if machine_load:
        max_machine_load = max(machine_load.values())
    else:
        max_machine_load = 1.0

    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        vals = list(opts.values())
        min_dur = min(vals)
        total_dur = sum(vals)
        n_machines = len(vals)

        # Flexibility: ratio of min to total duration
        # High flexibility = can run on many machines with similar times
        # This is the key contention-avoidance signal
        flexibility = min_dur / (total_dur + 1e-9)

        # Contention: inverse of normalized minimum load
        # Prefer operations whose best machine has the least current load
        min_load = min(machine_load[m] for m in opts)
        # Normalize to [0,1] range where 1 = least contention
        contention_score = 1.0 - (min_load / (max_machine_load + 1e-9))

        # Remaining work urgency (weaker signal)
        rw = op["remaining_work"]
        ro = op["remaining_ops"]

        # Composite score: primarily flexibility (contention avoidance),
        # secondary contention score, tertiary urgency
        # Weight: flexibility dominates (0.5), contention (0.3), urgency (0.2)
        urgency_norm = rw / (max_machine_load + rw + 1e-9) if ro > 0 else 1.0

        score = 0.5 * flexibility + 0.3 * contention_score + 0.2 * urgency_norm

        # Tiny tie-breaker
        score += 1e-8 * rw

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
