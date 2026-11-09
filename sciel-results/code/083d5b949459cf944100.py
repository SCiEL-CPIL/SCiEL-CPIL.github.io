"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Hard wall-clock guard: the evaluator kills at 150s; keep a safety margin.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 145.0  # seconds


def _deadline_exceeded():
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    # Safety guard: if we are about to exceed the wall-clock budget,
    # fall back to a cheap deterministic rule immediately.
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # --- Compute per-operation scores ---
    # We use a composite heuristic that balances:
    #   1. Remaining work (critical-path pressure) – higher is more urgent
    #   2. Minimum processing time among eligible machines – shorter ops reduce contention
    #   3. Number of remaining operations (job-level urgency)
    #   4. Quadratic penalty on the minimum duration to smooth the surrogate

    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each ready operation, find the minimum duration across its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) for op in ready_operations
    ], dtype=np.float64)

    # Normalize features to [0, 1] range for stable weighting
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    nw_norm = _normalize(remaining_work)
    ro_norm = _normalize(remaining_ops)
    md_norm = _normalize(min_durations)

    # Quadratic penalty on minimum duration: favors shorter operations more strongly
    # This acts as a smooth surrogate for the nonsmooth "pick the shortest" rule
    quad_penalty = md_norm ** 2

    # Composite score: higher is better (we pick the max)
    # Weights: emphasize critical path (remaining_work), penalize long durations
    w_rw = 0.60
    w_ro = 0.30
    w_md = 0.10

    scores = w_rw * nw_norm + w_ro * ro_norm - w_md * quad_penalty

    best_idx = int(np.argmax(scores))

    # Optionally choose the best machine (shortest duration) for that operation
    best_op = ready_operations[best_idx]
    best_machine = min(best_op["machine_options"], key=lambda m: best_op["machine_options"][m])

    # Return (index, machine) to let the simulator use the optimal machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
