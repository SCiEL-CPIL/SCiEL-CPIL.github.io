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

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        remaining_work[i] = op.get("remaining_work", 0)
        remaining_ops[i] = op.get("remaining_ops", 0)
        if mo:
            bm = min(mo, key=mo.get)
            best_machines[i] = bm
            min_dur[i] = mo[bm]
        else:
            min_dur[i] = 0.0

    # Guard: if we are dangerously close to the deadline, fall back to a
    # cheap, deterministic choice (max remaining work, tie-break min duration).
    if _elapsed() > _DEADLINE - 5.0:
        idx = int(np.argmax(remaining_work - 0.0))
        # deterministic tie-break
        best = remaining_work[idx]
        for i in range(n):
            if remaining_work[i] > best + 1e-9:
                idx = i
                best = remaining_work[i]
        if best_machines[idx] is not None:
            return (idx, best_machines[idx])
        return int(idx)

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
    dur_norm = _norm(min_dur)                 # longer processing -> less urgent

    # Efficiency ratio: remaining_work / min_dur
    # Higher ratio means the operation has a lot of work but can be done quickly
    # on its best machine -> high priority (reduces makespan effectively).
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.where(min_dur > 1e-9, remaining_work / min_dur, 0.0)
    eff_norm = _norm(efficiency)

    # Composite urgency score.
    # Primary: efficiency (work-to-min-duration ratio) — captures "can be done fast".
    # Secondary: remaining work (criticality).
    # Tertiary: shortest processing time (SPT) to free the machine sooner.
    # Quaternary: tail-latency reduction by favoring jobs with fewer remaining ops.
    w_eff = 1.2
    w_work = 0.8
    w_dur = 0.4
    w_ops = 0.15

    scores = w_eff * eff_norm + w_work * work_norm - w_dur * dur_norm - w_ops * ops_norm

    # Deterministic tie-break: among (near-)ties, prefer fewer remaining ops,
    # then lower index.
    best_idx = int(np.argmax(scores))
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        best_idx = min(tied, key=lambda i: (remaining_ops[i], i))

    # Machine selection: pick the machine that minimizes the resulting processing
    # time for the chosen operation. This aligns with the finding that processing-
    # time-driven machine selection outperforms load-based heuristics when we don't
    # receive completion events (load estimates are poor proxies for availability).
    mo = ready_operations[best_idx].get("machine_options", {})
    if mo:
        # Select machine with minimum duration; tie-break by lower machine key
        # for determinism.
        chosen_machine = min(mo, key=lambda m: (mo[m], m))
        return (int(best_idx), chosen_machine)
    return int(best_idx)
# EVOLVE-BLOCK-END
