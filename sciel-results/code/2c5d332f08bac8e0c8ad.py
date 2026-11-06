"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150s wall-clock.
# Keep a safety margin and, if we approach it, return a cheap deterministic
# choice immediately.
_DEADLINE = 145.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()
    return _time.monotonic() - _start_time


def _softmin(v, beta=8.0):
    """Smooth min surrogate via log-sum-exp (stable). Higher beta -> closer to true min."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return 0.0
    m = v.min()
    return m - np.log(np.sum(np.exp(-beta * (v - m)))) / beta


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        mo = ready_operations[0].get("machine_options", {})
        if mo:
            bm = min(mo, key=mo.get)
            return (0, bm)
        return 0

    # Precompute per-operation features.
    remaining_work = np.empty(n)
    remaining_ops = np.empty(n)
    min_dur = np.empty(n)
    max_dur = np.empty(n)
    num_machines = np.empty(n)
    tight_ratio = np.empty(n)      # min/max: high = little slack (tight)
    best_machines = [None] * n
    machine_lists = [None] * n

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        remaining_work[i] = op.get("remaining_work", 0)
        remaining_ops[i] = op.get("remaining_ops", 0)
        if mo:
            vals = np.array(list(mo.values()), dtype=float)
            keys = list(mo.keys())
            min_dur[i] = vals.min()
            max_dur[i] = vals.max()
            num_machines[i] = len(mo)
            bm = keys[int(np.argmin(vals))]
            best_machines[i] = bm
            machine_lists[i] = (keys, vals)
            # tightness: few options AND little slack (min close to max)
            tight_ratio[i] = (vals.min() / vals.max()) if vals.max() > 1e-12 else 1.0
        else:
            min_dur[i] = 0.0
            max_dur[i] = 0.0
            num_machines[i] = 0
            tight_ratio[i] = 0.0
            best_machines[i] = None
            machine_lists[i] = (None, None)

    # Deadline guard: cheap deterministic choice (constraint-tight first, then work).
    if _elapsed() > _DEADLINE - 5.0:
        idx = int(np.argmin(num_machines))
        if best_machines[idx] is not None:
            return (idx, best_machines[idx])
        return int(idx)

    # Normalize features to [0,1].
    def _norm(v):
        lo = v.min()
        hi = v.max()
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    # Constraint-tightness features (higher = more urgent to secure now).
    # Fewer machine options = tighter. Less slack (min/max high) = tighter.
    tightness = _norm(-num_machines) * 0.5 + _norm(tight_ratio) * 0.5

    # Work-load and SPT features.
    work_norm = _norm(remaining_work)
    # Smooth SPT proxy (soft-min over machine durations) for robustness.
    spt = np.array([_softmin(mv) if ml is not None else 0.0 for ml, mv in
                    [(m, v) for m, v in machine_lists]])
    spt_norm = _norm(spt)
    ops_norm = _norm(remaining_ops)

    # Composite urgency: constraint tightness primary, work secondary, SPT tertiary.
    w_tight = 1.0
    w_work = 0.5
    w_spt = 0.3
    w_ops = 0.1

    scores = (w_tight * tightness
              + w_work * work_norm
              - w_spt * spt_norm
              - w_ops * ops_norm)

    best_idx = int(np.argmax(scores))
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        best_idx = min(tied, key=lambda i: (num_machines[i], -tight_ratio[i],
                                            remaining_ops[i], min_dur[i], i))

    # Machine selection: minimum projected idle gap => smallest duration;
    # break ties deterministically by machine key.
    ml = machine_lists[best_idx]
    if ml is not None:
        keys, vals = ml
        if len(keys) == 1:
            return (int(best_idx), keys[0])
        vmin = vals.min()
        # among near-min-duration machines, pick the one with least idle gap
        # (smallest duration) then deterministic key order.
        cands = [keys[j] for j in range(len(keys)) if abs(vals[j] - vmin) < 1e-9]
        cands.sort()
        return (int(best_idx), cands[0])
    return int(best_idx)
# EVOLVE-BLOCK-END
