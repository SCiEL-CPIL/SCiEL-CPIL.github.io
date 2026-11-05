"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = _time.time() + 145.0  # safety margin under the 150s kill
_EPS = 1e-9


def dispatch(ready_operations, time):
    # Deadline guard: return a safe deterministic choice if we are running out of time.
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-extract fields once for speed.
    min_durs = [min(op["machine_options"].values()) for op in ready_operations]
    rem_work = [op["remaining_work"] for op in ready_operations]
    rem_ops = [op["remaining_ops"] for op in ready_operations]

    # Scale factors to normalize the two signals to comparable magnitudes.
    # SPT component: prefer smaller min duration (frees machine sooner).
    # MWRT component: prefer larger remaining work (progress on big jobs).
    max_min_dur = max(min_durs) if min_durs else 1.0
    max_rem_work = max(rem_work) if rem_work else 1.0
    if max_min_dur < _EPS:
        max_min_dur = 1.0
    if max_rem_work < _EPS:
        max_rem_work = 1.0

    # NSPT-style blend: score = w_spt * (1 - min_dur/max_min_dur) + w_mwrt * (rem_work/max_rem_work)
    # Higher score is better. Weights chosen to balance machine-freeing vs job-progress.
    w_spt = 0.5
    w_mwrt = 0.5

    best_score = -1.0
    best_idx = 0
    best_min_dur = min_durs[0]
    best_rem_ops = rem_ops[0]

    for i in range(n):
        spt_part = 1.0 - (min_durs[i] / max_min_dur)
        mwrt_part = rem_work[i] / max_rem_work
        score = w_spt * spt_part + w_mwrt * mwrt_part

        # Tie-break: prefer operation that frees the machine sooner (smaller min_dur),
        # then prefer fewer remaining ops (finishes job sooner to release downstream work).
        if score > best_score + _EPS:
            best_score = score
            best_idx = i
            best_min_dur = min_durs[i]
            best_rem_ops = rem_ops[i]
        elif abs(score - best_score) <= _EPS:
            if min_durs[i] < best_min_dur - _EPS or (
                abs(min_durs[i] - best_min_dur) <= _EPS and rem_ops[i] < best_rem_ops
            ):
                best_idx = i
                best_min_dur = min_durs[i]
                best_rem_ops = rem_ops[i]

    return int(best_idx)
# EVOLVE-BLOCK-END
