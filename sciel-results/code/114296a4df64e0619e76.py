"""Job-Shop dispatch: composite critical-ratio + SPT + machine-contention score."""
import time as _time
import numpy as np

# Deadline guard: keep a safety margin well under the 150s wall-clock kill.
_DEADLINE = _time.time() + 130.0


def _pick_best_machine(machine_options):
    """Return (best_machine, best_duration, min_duration, max_duration, spread)."""
    best_machine = None
    best_dur = None
    mn = None
    mx = None
    for m, d in machine_options.items():
        d = float(d)
        if best_dur is None or d < best_dur:
            best_dur = d
            best_machine = m
        if mn is None or d < mn:
            mn = d
        if mx is None or d > mx:
            mx = d
    spread = (mx - mn) if (mn is not None and mx is not None) else 0.0
    return best_machine, best_dur, mn, mx, spread


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Build per-op features.
    # machine contention: count how many ready ops target each machine (best machine).
    machine_count = {}
    best_machine_list = [None] * n
    best_dur_list = [0.0] * n
    crit_list = [0.0] * n
    spt_list = [0.0] * n

    # Pass 1: compute best machine + critical ratio.
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        bm, bd, mn, mx, spread = _pick_best_machine(mo)
        best_machine_list[i] = bm
        best_dur_list[i] = bd
        # critical ratio: remaining work per remaining op (urgency).
        rem_ops = op.get("remaining_ops", 1)
        rem_work = op.get("remaining_work", 0.0)
        if rem_ops and rem_ops > 0:
            crit = float(rem_work) / float(rem_ops)
        else:
            crit = float(rem_work)
        crit_list[i] = crit

    for i in range(n):
        m = best_machine_list[i]
        machine_count[m] = machine_count.get(m, 0) + 1

    # Determine scales for normalization.
    crit_arr = np.array(crit_list, dtype=float)
    dur_arr = np.array(best_dur_list, dtype=float)
    c_min, c_max = float(np.min(crit_arr)), float(np.max(crit_arr))
    d_min, d_max = float(np.min(dur_arr)), float(np.max(dur_arr))
    c_range = (c_max - c_min) if c_max > c_min else 1.0
    d_range = (d_max - d_min) if d_max > d_min else 1.0

    # Weights: emphasize critical ratio, then SPT (shorter is better), then contention.
    W_CRIT = 1.0
    W_SPT = 0.6
    W_CONT = 0.4

    best_idx = 0
    best_score = -1e18
    for i in range(n):
        # Normalized critical ratio (higher = more urgent).
        ncrit = (crit_list[i] - c_min) / c_range
        # Normalized SPT: shorter duration is better -> invert.
        nspt = 1.0 - (best_dur_list[i] - d_min) / d_range
        # Contention: fewer competitors on the same machine is better.
        cnt = machine_count[best_machine_list[i]]
        ncont = 1.0 / (1.0 + (cnt - 1))  # 1.0 if alone, decays with more competitors

        score = W_CRIT * ncrit + W_SPT * nspt + W_CONT * ncont
        if score > best_score:
            best_score = score
            best_idx = i

        # Deadline guard: bail early with best-so-far if we're close to the limit.
        if _time.time() > _DEADLINE - 10.0:
            break

    return int(best_idx)
