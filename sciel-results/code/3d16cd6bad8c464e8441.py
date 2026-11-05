"""Look-ahead critical-path dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

# Wall-clock budget guard: the evaluator kills the process at ~150s.
# We keep an explicit deadline with a safety margin and return the best
# decision found before it.
_DEADLINE = _time.monotonic() + 145.0  # 5s safety margin


def _critical_path(ready_operations):
    """Global makespan lower bound = max remaining work across all jobs."""
    max_rw = 0.0
    for op in ready_operations:
        rw = op["remaining_work"]
        if rw > max_rw:
            max_rw = rw
    return max_rw


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we're already close to the limit, fall back fast.
    if _time.monotonic() > _DEADLINE:
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))

    # Baseline critical path (max remaining work over all jobs).
    base_cp = _critical_path(ready_operations)

    # For each ready operation, simulate its immediate completion and compute
    # the resulting critical path. We want the operation whose completion
    # yields the greatest reduction in the global makespan lower bound.
    best_index = 0
    best_reduction = -1.0
    best_cp = base_cp

    for i in range(n):
        op = ready_operations[i]
        job = op["job"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]

        # Simulate completing this operation:
        # - remaining_work decreases by this op's duration (on its chosen machine)
        # - remaining_ops decreases by 1
        # We consider the machine that minimizes the job's new remaining work
        # (i.e., the shortest duration), since that best reduces the critical path.
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0.0

        new_job_remaining_work = remaining_work - min_dur
        if new_job_remaining_work < 0.0:
            new_job_remaining_work = 0.0

        # Compute the new critical path: max over all jobs of their (possibly
        # updated) remaining work. Only this job's remaining work changes.
        # We need the max remaining work of all OTHER jobs, then compare with
        # this job's new remaining work.
        other_max = 0.0
        for j in range(n):
            if j == i:
                continue
            rw = ready_operations[j]["remaining_work"]
            if rw > other_max:
                other_max = rw

        new_cp = new_job_remaining_work if new_job_remaining_work > other_max else other_max
        reduction = base_cp - new_cp

        # Prefer greater reduction; tie-break by smaller remaining_ops (finishes job sooner),
        # then by smaller remaining_work, then by index for determinism.
        if (reduction > best_reduction + 1e-9 or
            (abs(reduction - best_reduction) <= 1e-9 and
             (remaining_ops < ready_operations[best_index]["remaining_ops"] or
              (remaining_ops == ready_operations[best_index]["remaining_ops"] and
               remaining_work < ready_operations[best_index]["remaining_work"])))):
            best_reduction = reduction
            best_cp = new_cp
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
