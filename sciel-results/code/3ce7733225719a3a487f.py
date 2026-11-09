"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and which machine).
    
    Strategy: 
    - Use a composite heuristic that balances:
      1. Critical path priority: jobs with more remaining work should be prioritized
         to reduce makespan (Longest Remaining Processing Time - LRPT).
      2. Machine load balancing: prefer machines that are less utilized.
      3. Processing time: shorter operations can be completed sooner, freeing up machines.
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
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

    # For each ready operation, compute:
    # - min_duration: shortest processing time among available machines
    # - remaining_work: total remaining work for the job
    # - remaining_ops: number of remaining operations for the job
    # - job: job identifier
    
    candidates = []
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Choose machine that minimizes duration; tie-break by machine id for determinism
        min_machine = None
        min_dur = float('inf')
        for m, d in machine_options.items():
            if d < min_dur or (d == min_dur and (min_machine is None or m < min_machine)):
                min_dur = d
                min_machine = m

        if min_machine is None:
            continue

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        job = op.get("job", 0)

        candidates.append({
            'index': i,
            'machine': min_machine,
            'min_dur': min_dur,
            'remaining_work': remaining_work,
            'remaining_ops': remaining_ops,
            'job': job
        })

    if not candidates:
        return 0

    if len(candidates) == 1:
        c = candidates[0]
        return (c['index'], c['machine'])

    # Smooth scoring:
    # - Use a soft-max over remaining work to avoid hard flips between close jobs.
    # - Use an exponential decay penalty for longer processing times.
    # - Add a small urgency term for more remaining operations.

    # Soft-max temperature for remaining work
    # Larger tau -> smoother; smaller tau -> closer to hard max
    tau = 5.0

    rw_values = [float(c['remaining_work']) for c in candidates]
    max_rw = max(rw_values) if rw_values else 0.0
    if max_rw <= 0:
        max_rw = 1.0

    # log-sum-exp for stability
    shifted = [rw / max_rw for rw in rw_values]
    max_shifted = max(shifted)
    exp_shifted = [s - max_shifted for s in shifted]
    # exp of shifted values (bounded)
    exp_vals = [_math.exp(s) for s in exp_shifted]
    sum_exp = sum(exp_vals)
    if sum_exp == 0:
        sum_exp = 1.0

    # Duration penalty scale: shorter durations get higher score
    # Use exponential decay: score = exp(-dur / scale)
    # Choose scale based on typical duration to keep values in [~0.1, 1]
    durs = [float(c['min_dur']) for c in candidates]
    avg_d = sum(durs) / len(durs) if durs else 1.0
    if avg_d <= 0:
        avg_d = 1.0
    dur_scale = max(avg_d, 1.0)

    # Remaining ops normalization for urgency
    ro_values = [float(c['remaining_ops']) for c in candidates]
    max_ro = max(ro_values) if ro_values else 0.0
    if max_ro <= 0:
        max_ro = 1.0

    best_score = -float('inf')
    best_candidate = candidates[0]

    for idx, c in enumerate(candidates):
        # Soft-maxed remaining work contribution
        rw_score = exp_vals[idx] / sum_exp

        # Duration contribution
        dur_score = _math.exp(-c['min_dur'] / dur_scale)

        # Urgency contribution
        ro_score = c['remaining_ops'] / max_ro

        # Work-to-time ratio: how much remaining work per unit of processing time
        # Higher ratio = more critical job (more work to do relative to current op length)
        if c['min_dur'] > 0:
            work_time_ratio = c['remaining_work'] / c['min_dur']
        else:
            work_time_ratio = 0.0

        # Normalize work-time ratio for scoring
        max_wtr = max(float(cc['remaining_work'] / cc['min_dur']) if cc['min_dur'] > 0 else 0.0 for cc in candidates)
        if max_wtr <= 0:
            max_wtr = 1.0
        wtr_score = work_time_ratio / max_wtr

        # Composite score: emphasize critical path, then work-time urgency, then efficiency
        score = 0.50 * rw_score + 0.30 * wtr_score + 0.20 * dur_score

        if score > best_score:
            best_score = score
            best_candidate = c

    return (best_candidate['index'], best_candidate['machine'])
# EVOLVE-BLOCK-END
