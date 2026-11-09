"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and which machine).
    
    Strategy: 
    - Composite heuristic balancing Job Criticality (Remaining Work) and 
      Operation Efficiency (Work-to-Time Ratio).
    - Machine selection: Shortest Processing Time (SPT).
    
    Returns: (index, machine) tuple.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0

    candidates = []
    max_rw = 0.0
    max_wtr = 0.0
    max_ro = 0.0

    # Pass 1: Identify best machine and raw metrics
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        min_machine = None
        min_dur = float('inf')
        for m, d in machine_options.items():
            if d < min_dur or (d == min_dur and (min_machine is None or m < min_machine)):
                min_dur = d
                min_machine = m

        if min_machine is None:
            continue

        remaining_work = float(op.get("remaining_work", 0))
        remaining_ops = float(op.get("remaining_ops", 0))
        
        # Work-to-Time Ratio
        if min_dur > 0:
            wtr = remaining_work / min_dur
        else:
            wtr = 0.0

        candidates.append({
            'index': i,
            'machine': min_machine,
            'min_dur': min_dur,
            'rw': remaining_work,
            'ro': remaining_ops,
            'wtr': wtr
        })

        if remaining_work > max_rw:
            max_rw = remaining_work
        if wtr > max_wtr:
            max_wtr = wtr
        if remaining_ops > max_ro:
            max_ro = remaining_ops

    if not candidates:
        return 0

    if len(candidates) == 1:
        c = candidates[0]
        return (c['index'], c['machine'])

    # Handle zero maxes for normalization
    if max_rw <= 0: max_rw = 1.0
    if max_wtr <= 0: max_wtr = 1.0
    if max_ro <= 0: max_ro = 1.0

    best_score = -float('inf')
    best_candidate = candidates[0]

    # Weights: Emphasize Criticality (RW) and Efficiency (WTR)
    # WTR already includes RW, so it's a strong signal.
    # RO adds a tie-breaker for complexity.
    w_rw = 0.50
    w_wtr = 0.40
    w_ro = 0.10

    for c in candidates:
        rw_score = c['rw'] / max_rw
        wtr_score = c['wtr'] / max_wtr
        ro_score = c['ro'] / max_ro

        score = w_rw * rw_score + w_wtr * wtr_score + w_ro * ro_score

        if score > best_score:
            best_score = score
            best_candidate = c

    return (best_candidate['index'], best_candidate['machine'])
# EVOLVE-BLOCK-END
