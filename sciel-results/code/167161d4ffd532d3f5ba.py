"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a Critical Ratio (CR) dispatch rule with machine congestion awareness:
      1. CR = remaining_work / (current_time + estimated_completion_time)
         Lower CR means more urgent (less slack relative to work remaining).
      2. Machine selection: prefer the machine that minimizes the resulting
         makespan contribution, accounting for both duration and remaining ops.
      3. Tie-breaking: prefer jobs with more remaining operations (longer chains).
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Collect data for each ready operation
    best_machines = []
    best_durations = []
    remaining_works = []
    remaining_ops = []
    all_machine_durations = []  # list of dicts: machine -> duration
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
            all_machine_durations.append(machine_options)
        else:
            best_machine = None
            best_dur = 0
            all_machine_durations.append({})
        
        best_machines.append(best_machine)
        best_durations.append(best_dur)
        remaining_works.append(mw)
        remaining_ops.append(ro)
    
    # Critical Ratio: lower is more urgent
    # CR_i = remaining_work_i / (time + best_duration_i + avg_remaining_op_duration)
    # We estimate avg remaining op duration from remaining_work / remaining_ops
    
    best_idx = 0
    best_cr = float('inf')
    best_tiebreak = float('-inf')
    
    for i in range(n):
        rw = remaining_works[i]
        dur = best_durations[i]
        ro = remaining_ops[i]
        
        # Estimate average remaining operation duration
        if ro > 0:
            avg_remaining_dur = rw / ro
        else:
            avg_remaining_dur = dur
        
        # Time to complete this op + estimated time for remaining ops
        estimated_completion = dur + avg_remaining_dur * max(0, ro - 1)
        
        # Critical Ratio: lower means more urgent
        if time + estimated_completion > 0:
            cr = rw / (time + estimated_completion)
        else:
            cr = float('inf')
        
        # Tie-breaking: prefer more remaining ops (longer job chain)
        tiebreak = float(ro)
        
        if cr < best_cr or (cr == best_cr and tiebreak > best_tiebreak):
            best_cr = cr
            best_tiebreak = tiebreak
            best_idx = i
    
    # Machine selection: for the chosen operation, pick the machine that
    # minimizes a congestion-aware score. We prefer shorter durations but
    # also consider that machines with many options might be busier.
    # Simple approach: pick the machine with the shortest duration,
    # but if there's a tie or near-tie, prefer the one that appears
    # less frequently across all ready operations (less congested).
    
    chosen_op = ready_operations[best_idx]
    machine_options = chosen_op.get("machine_options", {})
    
    if machine_options:
        # Count how often each machine appears as an option across all ready ops
        machine_frequency = {}
        for op in ready_operations:
            for m in op.get("machine_options", {}).keys():
                machine_frequency[m] = machine_frequency.get(m, 0) + 1
        
        # Score each machine: primary = duration, secondary = negative frequency
        best_machine = None
        best_machine_score = float('inf')
        for m, d in machine_options.items():
            freq = machine_frequency.get(m, 0)
            # Lower duration is better; lower frequency is better (less congestion)
            # Combine: duration + small penalty for frequency
            machine_score = d + 0.01 * freq
            if machine_score < best_machine_score:
                best_machine_score = machine_score
                best_machine = m
        
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
