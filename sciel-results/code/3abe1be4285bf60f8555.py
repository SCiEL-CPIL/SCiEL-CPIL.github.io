"""Open-shop dispatch rule: CPM-proxy with smooth machine load balancing."""
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
    job_ids = []

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m, best_d = _best_machine_for_op(op)
        best_machines.append(best_m)
        min_durs.append(best_d)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        avg_durs.append(np.mean(list(opts.values())))
        scarcity_scores.append(0)
        job_ids.append(op["job"])

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

    # CPM proxy: prioritize operations on jobs with the longest remaining work
    # This is a direct measure of the "tail" or critical path length.
    
    # For each job, compute the maximum remaining work across all its ready ops
    # (in case multiple ops of the same job are ready)
    job_max_rem_work = {}
    for i in range(n):
        j = job_ids[i]
        rw = rem_works[i]
        if j not in job_max_rem_work or rw > job_max_rem_work[j]:
            job_max_rem_work[j] = rw

    # Assign each op the max remaining work of its job
    job_tails = []
    for i in range(n):
        job_tails.append(job_max_rem_work[job_ids[i]])

    # Normalize criteria
    def _normalize(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_tail = _normalize(job_tails)          # higher = longer critical path
    norm_scarcity = _normalize(scarcity_scores)  # higher = more contended
    norm_avg_dur = _normalize(avg_durs)          # higher = slower on avg
    norm_min_dur = _normalize(min_durs)          # higher = slower min duration

    # Weights for CPM-proxy strategy:
    # 1. Job tail (CPM): dominant factor, prioritize longest remaining jobs
    # 2. Machine scarcity: avoid bottlenecking contended machines
    # 3. Min duration: among tied tails, prefer faster ops
    # 4. Avg duration: secondary speed consideration
    
    w_tail = 0.50
    w_scarcity = 0.25
    w_min_dur = 0.15
    w_avg_dur = 0.10

    # Composite score: higher is better to schedule NOW
    scores = (
        w_tail * norm_tail
        + w_scarcity * (1.0 - norm_scarcity)
        + w_min_dur * (1.0 - norm_min_dur)
        + w_avg_dur * (1.0 - norm_avg_dur)
    )

    idx = int(np.argmax(scores))

    # Tie-break: if scores are very close, prefer lower min duration, then fewer rem_ops
    top_score = scores[idx]
    close_candidates = [i for i in range(n) if abs(scores[i] - top_score) < 1e-9]
    if len(close_candidates) > 1:
        idx = min(close_candidates, key=lambda i: (min_durs[i], rem_ops[i]))

    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
