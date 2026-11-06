"""Flexible job-shop dispatch: machine-aware tightness + successor-constraint lookahead."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    start_wall = _time.monotonic()
    deadline = start_wall + 0.5  # explicit guard; keep per-call cost tiny

    # 1) For each ready op, pick the eligible machine that is least busy.
    min_durs = [0.0] * n
    machine_loads = {}
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        best_load = None
        best_dur = None
        for m, d in mo.items():
            d = float(d)
            ml = float(mo.get(m + "_load", 0.0)) if isinstance(mo, dict) else 0.0
            # Prefer: earliest finish on that machine (load + duration), then shorter duration.
            finish = ml + d
            if best_load is None or finish < best_load or (finish == best_load and d < best_dur):
                best_load = finish
                best_dur = d
        min_durs[i] = best_dur if best_dur is not None else 1.0
        if best_dur is not None:
            machine_loads.setdefault(mo, None)  # placeholder, not used further

    # 2) Workload-weighted tightness: remaining_work / min_duration.
    tight = [0.0] * n
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        md = min_durs[i]
        tight[i] = rw / md if md > 0 else rw

    # 3) Successor-constraint lookahead: prefer ops whose next op has fewer machine
    #    options (more constrained) to avoid stranding jobs.
    succ_score = [0.0] * n
    for i, op in enumerate(ready_operations):
        rem = op.get("remaining_ops", 0)
        if rem <= 0:
            succ_score[i] = 0.0
        else:
            # If remaining_ops is a list of ops, use min machine-option count of next.
            ro = op.get("remaining_ops")
            if isinstance(ro, (list, tuple)) and len(ro) > 0:
                nxt = ro[0]
                if isinstance(nxt, dict):
                    mopts = nxt.get("machine_options", {})
                    cnt = len(mopts) if isinstance(mopts, dict) else 1
                    succ_score[i] = 1.0 / max(cnt, 1)
                else:
                    succ_score[i] = 0.0
            else:
                # Fallback: scale by remaining ops count (more remaining = slightly less urgent
                # for successor, but we want to start constrained ones; keep neutral-ish).
                succ_score[i] = 0.0

    # 4) Combine: primary tightness, secondary successor urgency, tertiary fewer options.
    best_i = 0
    best_key = None
    for i in range(n):
        op = ready_operations[i]
        mo = op["machine_options"]
        n_opts = len(mo) if isinstance(mo, dict) else 1
        key = (
            tight[i],
            succ_score[i],
            -float(n_opts),  # fewer options -> higher priority
        )
        if best_key is None or key > best_key:
            best_key = key
            best_i = i

    # Safety: if we're approaching the deadline, return immediately (already minimal work).
    if _time.monotonic() - start_wall > deadline:
        return int(best_i)

    return int(best_i)
# EVOLVE-BLOCK-END
