"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# We keep a safety margin and, if we ever approach it, return the best
# candidate found so far immediately.
_DEADLINE = 145.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()
    return _time.monotonic() - _start_time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        # Only one option; still try to specify its best machine.
        mo = ready_operations[0].get("machine_options", {})
        if mo:
            bm = min(mo, key=mo.get)
            return (0, bm)
        return 0

    # Precompute per-operation features.
    remaining_work = np.empty(n)
    remaining_ops = np.empty(n)
    min_dur = np.empty(n)
    best_machines = [None] * n
    max_dur = np.empty(n)
    num_machines = np.empty(n)
    soft_min_dur = np.empty(n)  # smooth surrogate for min duration

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        remaining_work[i] = op.get("remaining_work", 0)
        remaining_ops[i] = op.get("remaining_ops", 0)
        if mo:
            bm = min(mo, key=mo.get)
            best_machines[i] = bm
            min_dur[i] = mo[bm]
            max_dur[i] = max(mo.values())
            num_machines[i] = len(mo)
            # Soft-min over all machine durations: smoother than hard min
            durs = np.array(list(mo.values()), dtype=float)
            m = durs.min()
            soft_min_dur[i] = m - np.log(np.sum(np.exp(-8.0 * (durs - m)))) / 8.0
        else:
            min_dur[i] = 0.0
            max_dur[i] = 0.0
            num_machines[i] = 0
            soft_min_dur[i] = 0.0

    # Guard: if we are dangerously close to the deadline, fall back to a
    # cheap, deterministic choice (max remaining work, tie-break min duration).
    if _elapsed() > _DEADLINE - 5.0:
        idx = int(np.argmax(remaining_work))
        if best_machines[idx] is not None:
            return (idx, best_machines[idx])
        return int(idx)

    # Urgency ratio: remaining work per unit of effective processing time.
    # Higher ratio = more work cleared per time = more critical to schedule now.
    eps = 1e-9
    urgency_ratio = remaining_work / (soft_min_dur + eps)

    # Normalize features to [0,1] so weights are scale-invariant.
    def _norm(v):
        lo = v.min()
        hi = v.max()
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    # Normalized urgency components (higher = more urgent).
    work_norm = _norm(remaining_work)          # more remaining work -> more critical
    ops_norm = _norm(remaining_ops)           # more remaining ops -> less urgent
    dur_norm = _norm(soft_min_dur)            # longer effective processing -> less urgent
    urgency_norm = _norm(urgency_ratio)       # high work-per-time ratio -> more critical
    flexibility_norm = _norm(num_machines)    # more machine options -> more flexible

    # Composite urgency score.
    # Primary: urgency ratio (work per effective processing time) - bottleneck mitigation.
    # Secondary: remaining work (criticality).
    # Tertiary: shortest effective processing time to free the machine sooner.
    # Quaternary: tail-latency reduction by favoring jobs with fewer remaining ops.
    # Quinary: prefer operations with more flexibility (more machine options).
    w_urgency = 3.0
    w_work = 2.0
    w_dur = 0.5
    w_ops = 0.1
    w_flex = 0.05

    scores = w_urgency * urgency_norm + w_work * work_norm - w_dur * dur_norm - w_ops * ops_norm + w_flex * flexibility_norm

    # Deterministic tie-break: among (near-)ties, prefer higher urgency ratio,
    # then fewer remaining ops, then lower index.
    best_idx = int(np.argmax(scores))
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        best_idx = min(tied, key=lambda i: (-urgency_ratio[i], remaining_ops[i], min_dur[i], i))

    # Return (index, machine) to also specify machine assignment.
    if best_machines[best_idx] is not None:
        return (int(best_idx), best_machines[best_idx])
    return int(best_idx)
# EVOLVE-BLOCK-END
