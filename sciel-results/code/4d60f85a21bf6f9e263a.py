"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_KILL = _T0 + 112.0


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Composite heuristic:
    - Machine selection: minimize effective completion time considering
      local congestion (how many ready ops target the same machine) and
      a soft load-balancing penalty.
    - Operation selection: maximize urgency (remaining work / processing time)
      with a tail-clearing boost in the late phase, and a critical-path
      penalty for jobs with many remaining ops.
    """
    if _clock.time() > _KILL:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0

    # Collect features
    rw_list = []
    ro_list = []
    mo_list = []
    min_dur_list = []

    # Track per-machine demand among ready ops
    mach_demand = {}
    mach_dur_sum = {}

    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        rw_list.append(rw)
        ro_list.append(ro)
        mo_list.append(mo)

        if mo:
            bd = min(mo.values())
            min_dur_list.append(bd)
            for m, d in mo.items():
                mach_demand[m] = mach_demand.get(m, 0) + 1
                mach_dur_sum[m] = mach_dur_sum.get(m, 0) + d
        else:
            min_dur_list.append(0)

    if n == 1:
        mo = mo_list[0]
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return 0

    # Phase detection: fraction of remaining work that is short
    avg_rw = sum(rw_list) / n
    max_rw = max(rw_list) if rw_list else 1
    phase = avg_rw / (max_rw + 1e-9) if max_rw > 0 else 1.0

    # Global load estimate for normalization
    total_load = sum(mach_dur_sum.values()) if mach_dur_sum else 1
    if total_load == 0:
        total_load = 1

    scores = []
    chosen_machines = []

    for i in range(n):
        rw = rw_list[i]
        ro = ro_list[i]
        mo = mo_list[i]

        if not mo:
            chosen_machines.append(None)
            scores.append(0.0)
            continue

        # --- Machine selection ---
        # Cost = duration + congestion_penalty + load_balance_penalty
        best_m = None
        best_cost = float("inf")
        best_dur = float("inf")

        for m, d in mo.items():
            cnt = mach_demand.get(m, 0)
            # Congestion: smooth increase with demand
            cong_pen = 0.25 * cnt / (1.0 + 0.4 * cnt)
            # Load balance: penalize machines that already have a lot of
            # ready-work assigned (proxy for future queue)
            load_pen = 0.15 * (mach_dur_sum.get(m, 0) / total_load)
            cost = d + cong_pen + load_pen

            if cost < best_cost:
                best_cost = cost
                best_m = m
                best_dur = d

        chosen_machines.append(best_m)

        # --- Operation scoring ---
        # Urgency: how much remaining work per unit processing time
        urgency = rw / (best_dur + 0.5)

        # Criticality: jobs with many remaining ops are on the critical path
        # Penalize them slightly less (we want to finish them, so higher score)
        crit_bonus = 1.0 + 0.15 * ro

        # Tail clearing: in late phase, short-duration ops are more valuable
        # because they free up machines quickly
        if phase < 0.5:
            tail_boost = 1.0 + 0.3 * (1.0 - best_dur / (avg_rw + 1.0))
        else:
            tail_boost = 1.0

        # Congestion penalty on the chosen machine
        cnt = mach_demand.get(best_m, 0)
        cont_pen = 1.0 / (1.0 + 0.5 * cnt)

        score = urgency * crit_bonus * tail_boost * cont_pen
        scores.append(score)

    # Pick best
    best_idx = 0
    best_s = scores[0]
    for i in range(1, n):
        if scores[i] > best_s:
            best_s = scores[i]
            best_idx = i

    bm = chosen_machines[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx
# EVOLVE-BLOCK-END
