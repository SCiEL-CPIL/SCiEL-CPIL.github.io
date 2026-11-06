"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _best_machine_for_op(op):
    """Return (machine, duration) for the fastest option of an operation."""
    opts = op["machine_options"]
    best_m = min(opts, key=lambda m: (opts[m], m))
    return best_m, opts[best_m]


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    # Pre-compute per-operation features
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    avg_durs = []
    scarcity_scores = []

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m, best_d = _best_machine_for_op(op)
        best_machines.append(best_m)
        min_durs.append(best_d)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        # Average duration across all machine options
        avg_durs.append(np.mean(list(opts.values())))
        # Scarcity: how many other ready ops also want this machine?
        # We approximate by counting how many ops have this as their best machine
        scarcity_scores.append(0)  # will compute after first pass

    # Compute machine contention: for each op, how many other ops also
    # prefer the same best machine
    machine_counts = {}
    for m in best_machines:
        machine_counts[m] = machine_counts.get(m, 0) + 1
    for i in range(n):
        scarcity_scores[i] = machine_counts[best_machines[i]]

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Multi-criteria weighted heuristic:
    # 1. Remaining work (LPT): prioritize jobs with more total work left
    # 2. Machine scarcity: avoid bottlenecking a contended machine
    # 3. Average processing time: prefer ops that are generally faster
    # 4. Number of remaining ops: fewer remaining ops = closer to completion

    # Normalize each criterion to [0, 1] range for fair weighting
    def _normalize(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_rem_work = _normalize(rem_works)       # higher = more work left
    norm_scarcity = _normalize(scarcity_scores)  # higher = more contended
    norm_avg_dur = _normalize(avg_durs)          # higher = slower on avg
    norm_rem_ops = _normalize(rem_ops)           # higher = more ops left

    # Weights: balance the criteria
    # Emphasize remaining work (LPT) most, then avoid scarce machines,
    # then prefer faster ops, then fewer remaining ops
    w_work = 0.45
    w_scarcity = 0.25
    w_speed = 0.20
    w_ops = 0.10

    # Composite score: higher is better to schedule NOW
    # For scarcity, we WANT LOW scarcity (less contended), so invert
    # For speed, we WANT LOW avg duration, so invert
    # For rem_ops, we WANT LOW (closer to done), so invert
    scores = (
        w_work * norm_rem_work
        + w_scarcity * (1.0 - norm_scarcity)
        + w_speed * (1.0 - norm_avg_dur)
        + w_ops * (1.0 - norm_rem_ops)
    )

    idx = int(np.argmax(scores))

    # Tie-break: if scores are very close, prefer lower min duration
    top_score = scores[idx]
    close_candidates = [i for i in range(n) if abs(scores[i] - top_score) < 1e-9]
    if len(close_candidates) > 1:
        idx = min(close_candidates, key=lambda i: (min_durs[i], rem_ops[i]))

    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
