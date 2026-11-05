"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time as _time

# ---- Tunable hyperparameters for the urgency + efficiency + load blend ----
W_REM_WORK = 1.0      # weight on remaining_work (urgency)
W_TAIL     = 0.5      # weight on remaining_ops (critical-path tail urgency)
W_EFF      = 0.4      # weight on efficiency boost (fastest machine)
W_LOAD     = 0.3      # weight on least-loaded machine (contention)
W_BOTTLE   = 0.7      # weight on look-ahead bottleneck (earliest possible completion)
W_COMPLET  = 0.25     # weight on completeness bonus (prefer finishing jobs)
LSE_BETA   = 8.0      # sharpness of the soft-min over machine durations


def _soft_min(durations):
    """Smooth (log-sum-exp) surrogate for min over a list of positive durations.
    Returns a value close to the min but differentiable/smooth in the durations."""
    if not durations:
        return 0.0
    if len(durations) == 1:
        return durations[0]
    # min = - (1/beta) * log(sum(exp(-beta * d)))
    neg = [-LSE_BETA * d for d in durations]
    m = max(neg)
    s = math.exp(m) + sum(math.exp(x - m) for x in neg[1:])
    return -(m + math.log(s)) / LSE_BETA


def _best_duration(op):
    """Fastest (minimum) duration among eligible machines for this operation."""
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    return min(opts.values())


def _least_loaded_machine(op):
    """Pick the machine with the smallest duration (least contention) for this op."""
    opts = op.get("machine_options", {})
    if not opts:
        return None
    # min by duration; tie-break by machine key for determinism
    best_machine = None
    best_dur = float("inf")
    for machine, dur in opts.items():
        if dur < best_dur:
            best_dur = dur
            best_machine = machine
    return best_machine


def _earliest_completion(op, current_time):
    """Estimate the earliest time this op could finish if started now on its
    best (shortest-duration) eligible machine.  This is a look-ahead bottleneck
    signal: a small value means the op can be completed quickly, reducing the
    chance it becomes a straggler."""
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    best_dur = min(opts.values())
    return current_time + best_dur


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        m = _least_loaded_machine(op)
        if m is not None:
            return (0, m)
        return 0

    # ---- Per-operation raw signals ----
    rem_works = []
    tails = []
    best_durs = []
    soft_durs = []
    earliest_comps = []
    completions = []  # fraction of job completed (higher = closer to done)
    for op in ready_operations:
        rem_works.append(float(op.get("remaining_work", 0.0)))
        tails.append(float(op.get("remaining_ops", 0.0)))
        bd = _best_duration(op)
        best_durs.append(bd)
        soft_durs.append(_soft_min(list(op.get("machine_options", {}).values())))
        ec = _earliest_completion(op, time)
        earliest_comps.append(ec)
        # completeness: 1 - remaining_ops / total_ops; we don't have total_ops
        # directly, so use remaining_work as a proxy: smaller remaining_work
        # relative to the max across ready ops means more complete.
        completions.append(1.0)  # placeholder, normalized below

    # ---- Normalize each signal to [0, 1] (higher = more urgent / better) ----
    def _norm(vals):
        lo = min(vals)
        hi = max(vals)
        span = hi - lo
        if span <= 1e-12:
            return [0.5] * len(vals)
        return [(v - lo) / span for v in vals]

    nw = _norm(rem_works)          # more remaining work -> more urgent
    nt = _norm(tails)              # more remaining ops -> more urgent
    # efficiency: faster (smaller duration) -> higher score
    eff_raw = [(-d if d < float("inf") else 0.0) for d in best_durs]
    ne = _norm(eff_raw)
    # load: smaller soft-min duration -> less loaded -> higher
    load_raw = [(-d if d < float("inf") else 0.0) for d in soft_durs]
    nl = _norm(load_raw)
    # bottleneck: smaller earliest completion time -> less likely to be straggler -> higher
    bottle_raw = [(-c if c < float("inf") else 0.0) for c in earliest_comps]
    nb = _norm(bottle_raw)
    # completeness: use remaining_work inversely; ops with less remaining work
    # are closer to completion and should be prioritized to avoid stragglers.
    # We already have nw (normalized remaining work, higher = more work).
    # Invert it: nc = 1 - nw gives higher score for ops with less remaining work.
    nc = [1.0 - nw[i] for i in range(n)]

    # ---- Combined urgency + efficiency + load + bottleneck + completeness score ----
    scores = [
        W_REM_WORK * nw[i]
        + W_TAIL * nt[i]
        + W_EFF * ne[i]
        + W_LOAD * nl[i]
        + W_BOTTLE * nb[i]
        + W_COMPLET * nc[i]
        for i in range(n)
    ]

    # ---- Pick the best index (tie-break: earlier ready op) ----
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-12:
            best_score = scores[i]
            best_idx = i

    # ---- Optionally return the least-loaded machine to cut contention ----
    chosen = ready_operations[best_idx]
    m = _least_loaded_machine(chosen)
    if m is not None:
        return (best_idx, m)
    return best_idx
# EVOLVE-BLOCK-END
