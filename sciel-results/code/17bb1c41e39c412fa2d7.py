"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

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
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Precompute per-op info
    # For each operation, consider all machine options.
    # In an open shop, each operation must be processed on exactly one machine
    # (or a subset). The "machine_options" dict maps machine -> duration.
    # We pick which operation to schedule next, and optionally which machine.

    # Heuristic scoring (refined):
    # 1. Primary: Longest Remaining Work (LRPT) for the job.
    #    Prioritizing jobs with more total remaining work helps balance the
    #    overall load and reduces the likelihood of a long tail in makespan.
    # 2. Secondary: Soft-min duration (smooth surrogate for min machine time).
    #    Uses log-sum-exp style aggregation to capture both the best option
    #    and the spread, favoring ops with a clearly superior machine choice.
    # 3. Tertiary: Machine contention penalty.
    #    Penalize ops whose best machine is also wanted by many other ready ops.
    # 4. Quaternary: Most Constrained Variable (MCV) / Flexibility.
    #    Prioritize operations with fewer machine options (less flexibility).
    # 5. Quinary: Fewer remaining operations (finish jobs sooner).
    # 6. Senary: Job ID for determinism.

    # Compute min duration, best machine, flexibility, and machine contention
    min_durs = []
    best_machines = []
    flexibilities = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        flexibilities.append(len(opts))

    # Count how many ready ops prefer each machine (contention)
    machine_count = {}
    for m in best_machines:
        machine_count[m] = machine_count.get(m, 0) + 1

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Refined Combined Load-Balancing, Contention, and Flexibility heuristic
    # Primary: -remaining_work (Higher work = Higher priority)
    # Secondary: soft_min_duration (smooth surrogate for best machine time)
    # Tertiary: machine contention (fewer competitors on best machine = better)
    # Quaternary: -flexibility (Fewer options = Higher priority, avoids deadlocks)
    # Quinary: remaining_ops (Fewer ops = Higher priority, completes jobs)
    # Senary: job (Lower ID = Higher priority, determinism)

    def score_op(i):
        op = ready_operations[i]
        # Negate remaining_work so that higher work sorts first (ascending sort)
        primary = -op["remaining_work"]
        
        # Soft-min: use a weighted combination of min and mean duration
        # to create a smooth surrogate that captures both the best option
        # and the overall quality of machine options
        opts = op["machine_options"]
        durs = list(opts.values())
        min_d = min(durs)
        mean_d = sum(durs) / len(durs)
        # Blend: 0.7 * min + 0.3 * mean gives a smooth measure
        # that rewards having a very good best option while penalizing
        # poor alternatives
        soft_min_d = 0.7 * min_d + 0.3 * mean_d
        secondary = soft_min_d
        
        # Machine contention: fewer ops wanting the same machine is better
        tertiary = machine_count.get(best_machines[i], 0)
        
        # Fewer machine options is better (less flexible, schedule now)
        quaternary = -flexibilities[i]
        # Fewer remaining ops is better
        quinary = op["remaining_ops"]
        # Lower job ID is better
        senary = op["job"]
        
        return (primary, secondary, tertiary, quaternary, quinary, senary)

    order = sorted(range(n), key=score_op)
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
