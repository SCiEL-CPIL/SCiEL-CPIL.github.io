import time as time_module
from collections import defaultdict

def _compute_paths(ready_operations):
    """Return (critical[job], minmach[(job, m)]) for all ready operations.

    critical[job]  = min over the job's ready ops of (min dur of that op
                  + sum of min dur of that job's later ready ops).
    minmach[(job, m)] = min duration of the job's ready op that uses machine m.
    """
    by_job = defaultdict(list)
    for op in ready_operations:
        by_job[op["job"]].append(op)

    critical = {}
    minmach = {}
    for job, ops in by_job.items():
        # sort by number of remaining ops (desc), ties by job id for determinism
        ops_sorted = sorted(ops, key=lambda o: (-len(o.get("remaining_ops") or []), o["job"]))
        crit = {}
        for idx, op in enumerate(ops_sorted):
            opts = op.get("machine_options") or {}
            durs = list(opts.values())
            min_dur = min(durs) if durs else 0
            tail = 0
            for later in ops_sorted[idx + 1:]:
                ld = list((later.get("machine_options") or {}).values())
                tail += min(ld) if ld else 0
            crit[id(op)] = min_dur + tail
            for m, d in opts.items():
                key = (job, m)
                if key not in minmach or d < minmach[key]:
                    minmach[key] = d
        # keep the best (min) critical path for the job
        critical[job] = min(crit.values()) if crit else 0
    return critical, minmach


def _makespan_bound(ready_operations, time, skip_idx=None):
    """Lower bound on makespan from `time` after starting op `skip_idx`.

    bound = time + max( max_j C_j, max_m M_m )
    where C_j is job j's remaining critical path and M_m is machine m's
    remaining workload lower bound, both excluding the started operation.
    """
    if skip_idx is None:
        remaining = list(ready_operations)
    else:
        remaining = [op for i, op in enumerate(ready_operations) if i != skip_idx]
    if not remaining:
        return time

    critical, minmach = _compute_paths(remaining)

    max_c = max(critical.values()) if critical else 0

    per_mach = defaultdict(float)
    for op in remaining:
        for m, d in (op.get("machine_options") or {}).items():
            per_mach[m] += d
    max_m = max(per_mach.values()) if per_mach else 0

    return time + max(max_c, max_m)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        return 0

    deadline = time_module.time() + 145.0  # safety margin before kill

    n = len(ready_operations)
    best_idx = 0
    best_bound = None
    for i in range(n):
        if time_module.time() > deadline:
            break
        b = _makespan_bound(ready_operations, time, skip_idx=i)
        if best_bound is None or b < best_bound:
            best_bound = b
            best_idx = i

    return best_idx
