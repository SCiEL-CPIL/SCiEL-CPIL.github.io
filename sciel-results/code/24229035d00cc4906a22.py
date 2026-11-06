import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
_START_TIME = _time.monotonic()
_DEADLINE_MARGIN = 5.0  # seconds

def _elapsed():
    return _time.monotonic() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0  # Fallback, shouldn't happen

    # Check deadline
    if _deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        mo = ready_operations[best_idx].get('machine_options', {})
        if mo:
            bm = min(mo, key=mo.get)
            return best_idx, bm
        return best_idx, 0

    n = len(ready_operations)

    # Precompute per-operation features
    remaining_work = np.empty(n)
    remaining_ops = np.empty(n)
    min_dur = np.empty(n)
    best_machines = [None] * n
    num_machines = np.empty(n)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        remaining_work[i] = op.get("remaining_work", 0)
        remaining_ops[i] = op.get("remaining_ops", 0)
        if mo:
            bm = min(mo, key=mo.get)
            best_machines[i] = bm
            min_dur[i] = mo[bm]
            num_machines[i] = len(mo)
        else:
            min_dur[i] = 0.0
            num_machines[i] = 0

    # Handle edge cases
    if n == 1:
        mo = ready_operations[0].get("machine_options", {})
        if mo:
            bm = min(mo, key=mo.get)
            return (0, bm)
        return 0, 0

    # Normalize features to [0,1] so weights are scale-invariant
    def _norm(v):
        lo = v.min()
        hi = v.max()
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    # Normalized urgency components (higher = more urgent)
    work_norm = _norm(remaining_work)          # more remaining work -> more critical
    ops_norm = _norm(remaining_ops)           # more remaining ops -> less urgent
    dur_norm = _norm(min_dur)                 # longer processing -> less urgent
    flexibility_norm = _norm(num_machines)    # more machine options -> more flexible

    # Constraint tightness: fewer machine options = higher urgency
    # Use 1/(num_machines + 0.5) as a constraint tightness proxy
    constraint = 1.0 / (num_machines + 0.5)
    constraint_norm = _norm(constraint)

    # Composite urgency score.
    # Primary: remaining work (criticality)
    # Secondary: constraint tightness (fewer options = more urgent)
    # Tertiary: shortest processing time (SPT) to free the machine sooner
    # Quaternary: tail-latency reduction by favoring jobs with fewer remaining ops
    # Quinary: prefer operations with more flexibility (more machine options)
    w_work = 1.0
    w_constraint = 0.8
    w_dur = 0.4
    w_ops = 0.1
    w_flex = 0.05

    scores = (w_work * work_norm + 
              w_constraint * constraint_norm - 
              w_dur * dur_norm - 
              w_ops * ops_norm + 
              w_flex * flexibility_norm)

    # Deterministic tie-break: among (near-)ties, prefer fewer remaining ops,
    # then lower index.
    best_idx = int(np.argmax(scores))
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        best_idx = min(tied, key=lambda i: (remaining_ops[i], min_dur[i], i))

    # Machine selection: use the machine with minimum duration for the chosen operation
    # This is a simple SPT-based machine choice
    mo = ready_operations[best_idx].get("machine_options", {})
    if mo:
        best_machine = min(mo, key=mo.get)
    else:
        best_machine = 0

    return int(best_idx), best_machine
