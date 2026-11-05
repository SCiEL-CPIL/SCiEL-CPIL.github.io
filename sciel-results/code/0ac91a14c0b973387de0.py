"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 140.0  # safety margin before 150s kill
# per-machine accumulated load (heuristic proxy for future availability)
_MACHINE_LOAD = {}


def _elapsed():
    return _time.monotonic() - _START


def dispatch(ready_operations, time):
    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # fall back to the cheapest deterministic rule immediately.
    if _elapsed() > _DEADLINE - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Score each ready operation.
    # We prefer:
    #   (a) large remaining work (finish big jobs early, reduce tail),
    #   (b) few remaining ops (jobs close to completion shouldn't be starved),
    #   (c) a short, low-load machine assignment (SPT on least-loaded machine
    #       reduces congestion and keeps the critical path short).
    best_idx = 0
    best_score = -float("inf")

    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))

        # Best (shortest) machine option, tie-break by current load.
        mo = op.get("machine_options", {})
        if mo:
            best_dur = float("inf")
            best_load = float("inf")
            for m, d in mo.items():
                d = float(d)
                load = _MACHINE_LOAD.get(m, 0.0)
                if d < best_dur or (d == best_dur and load < best_load):
                    best_dur = d
                    best_load = load
            # SPT: shorter duration is better -> use -best_dur.
            # Load: less loaded machine is better -> use -best_load.
            machine_term = -best_dur - 0.5 * best_load
        else:
            machine_term = 0.0

        # Weighted composite. remaining_work dominates; machine choice is a
        # secondary tie-breaker that can flip close calls.
        score = rw + 0.1 * ro + 0.05 * machine_term

        if score > best_score:
            best_score = score
            best_idx = i

    # Update the load proxy for the chosen operation's best machine.
    mo = ready_operations[best_idx].get("machine_options", {})
    if mo:
        best_m = None
        best_dur = float("inf")
        best_load = float("inf")
        for m, d in mo.items():
            d = float(d)
            load = _MACHINE_LOAD.get(m, 0.0)
            if d < best_dur or (d == best_dur and load < best_load):
                best_dur = d
                best_load = load
                best_m = m
        if best_m is not None:
            _MACHINE_LOAD[best_m] = _MACHINE_LOAD.get(best_m, 0.0) + best_dur

    return int(best_idx)
# EVOLVE-BLOCK-END
