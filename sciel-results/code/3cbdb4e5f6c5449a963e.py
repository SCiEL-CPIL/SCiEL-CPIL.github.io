"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def _min_duration(op):
    """Minimum processing time among machine options."""
    if not op["machine_options"]:
        return 0.0
    return min(op["machine_options"].values())


def _max_duration(op):
    """Maximum processing time among machine options."""
    if not op["machine_options"]:
        return 0.0
    return max(op["machine_options"].values())


def _avg_duration(op):
    """Average processing time among machine options."""
    if not op["machine_options"]:
        return 0.0
    vals = list(op["machine_options"].values())
    return sum(vals) / len(vals)


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Returns: index of the ready operation to start, optionally (index, machine).
    
    Strategy: Combine multiple priority signals:
    1. Urgency: jobs with fewer remaining ops are more critical (closer to completion)
    2. Remaining work: total remaining processing time
    3. Efficiency: prefer operations that can be done quickly (shorter duration)
    4. Machine selection: pick the machine with shortest duration for the chosen op
    
    We use a weighted score that normalizes each component to [0,1] across
    ready operations, then combines them with weights tuned to balance
    urgency vs. efficiency.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        if op["machine_options"]:
            best_machine = min(op["machine_options"], key=op["machine_options"].get)
            return (0, best_machine)
        return 0

    # Precompute features for each ready operation
    remaining_works = []
    remaining_ops = []
    min_durs = []
    avg_durs = []
    max_durs = []

    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 1)
        mo = op["machine_options"]
        if mo:
            md = min(mo.values())
            ad = sum(mo.values()) / len(mo)
            xd = max(mo.values())
        else:
            md = 0.0
            ad = 0.0
            xd = 0.0
        remaining_works.append(rw)
        remaining_ops.append(ro)
        min_durs.append(md)
        avg_durs.append(ad)
        max_durs.append(xd)

    rw_arr = np.array(remaining_works, dtype=float)
    ro_arr = np.array(remaining_ops, dtype=float)
    md_arr = np.array(min_durs, dtype=float)
    ad_arr = np.array(avg_durs, dtype=float)

    # Normalize each feature to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    rw_norm = normalize(rw_arr)       # higher = more work remaining
    ro_norm = normalize(ro_arr)       # higher = more ops remaining (less urgent)
    md_norm = normalize(md_arr)       # higher = slower min duration

    # Urgency score: prioritize jobs with FEWER remaining ops and MORE remaining work
    # Invert remaining_ops so fewer ops = higher urgency
    urgency_from_ops = 1.0 - ro_norm          # fewer remaining ops -> higher
    urgency_from_work = rw_norm               # more work -> higher (critical path)

    # Efficiency score: prefer faster operations (lower duration)
    efficiency = 1.0 - md_norm                # shorter min duration -> higher

    # Combined score: weighted sum
    # Weights: urgency components dominate, efficiency is secondary
    w_urgency_ops = 0.45
    w_urgency_work = 0.35
    w_efficiency = 0.20

    scores = (w_urgency_ops * urgency_from_ops +
              w_urgency_work * urgency_from_work +
              w_efficiency * efficiency)

    best_idx = int(np.argmax(scores))

    # Select the machine with the shortest duration for the chosen operation
    op = ready_operations[best_idx]
    if op["machine_options"]:
        best_machine = min(op["machine_options"], key=op["machine_options"].get)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
