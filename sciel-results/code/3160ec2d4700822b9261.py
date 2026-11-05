"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

# Critical-path tracking: for each job, track the estimated remaining critical path length
_job_remaining_critical = {}  # job_id -> estimated remaining critical path length
_job_total_work = {}          # job_id -> total work (first seen)

def _reset_state():
    global _machine_load, _job_remaining_critical, _job_total_work
    _machine_load = np.zeros(_MACHINE_COUNT)
    _job_remaining_critical = {}
    _job_total_work = {}

def dispatch(ready_operations, time):
    """
    Lexicographic critical-path dispatch rule.
    
    Structural change: Instead of a smooth surrogate for bottleneck capacity,
    we directly model the makespan as the maximum over all job completion times.
    
    Priority is determined lexicographically:
    1. Primary: operations with the longest remaining critical path (most urgent for makespan)
    2. Secondary: operations that can be completed earliest (reduce current makespan contribution)
    3. Tertiary: operations with fewer machine options (more constrained)
    
    Machine selection: minimize the projected makespan increase, with a look-ahead
    penalty for machines that will be heavily loaded by other ready operations.
    """
    global _machine_load, _last_seen_time, _job_remaining_critical, _job_total_work

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance
    if time < _last_seen_time:
        _reset_state()
    _last_seen_time = time

    # Update job tracking: estimate remaining critical path per job
    for op in ready_operations:
        j = op["job"]
        rw = op["remaining_work"]
        if j not in _job_total_work:
            _job_total_work[j] = rw
        # The remaining critical path for a job is at least its remaining work
        # We use remaining_work as a proxy for critical path length
        if j not in _job_remaining_critical:
            _job_remaining_critical[j] = rw
        else:
            _job_remaining_critical[j] = min(_job_remaining_critical[j], rw)

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = None
            best_completion = float('inf')
            for m, d in machine_options.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_completion:
                    best_completion = completion
                    best_machine = m
            if best_machine is not None and best_machine < _MACHINE_COUNT:
                _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Compute lexicographic priority components for each operation
    # Key insight: makespan = max over jobs of (completion time of last op)
    # So we should prioritize jobs with the most remaining critical work,
    # and within that, operations that can be done soonest.
    
    # For each op, compute:
    # 1. job_critical: remaining critical path of the job (higher = more urgent)
    # 2. earliest_start: when this op can start on its best machine
    # 3. op_duration: processing time on best machine
    # 4. flexibility: number of machine options (fewer = more constrained)
    
    job_critical = np.zeros(n)
    earliest_starts = np.zeros(n)
    op_durations = np.zeros(n)
    flexibilities = np.zeros(n)
    best_machines = [None] * n
    best_completions = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        j = op["job"]
        mo = op["machine_options"]
        
        if not mo:
            job_critical[i] = -float('inf')
            earliest_starts[i] = float('inf')
            op_durations[i] = float('inf')
            flexibilities[i] = 0
            best_completions[i] = float('inf')
            continue
        
        # Job critical path: remaining work in the job
        job_critical[i] = _job_remaining_critical.get(j, op["remaining_work"])
        
        # Find best machine (earliest completion)
        best_comp = float('inf')
        best_m = None
        best_start = float('inf')
        best_dur = float('inf')
        for m, d in mo.items():
            if m < _MACHINE_COUNT:
                start = max(time, _machine_load[m])
                comp = start + d
            else:
                start = time
                comp = time + d
            if comp < best_comp:
                best_comp = comp
                best_m = m
                best_start = start
                best_dur = d
        
        earliest_starts[i] = best_start
        op_durations[i] = best_dur
        best_completions[i] = best_comp
        best_machines[i] = best_m
        flexibilities[i] = len(mo)

    # Lexicographic sorting:
    # Primary key: job_critical (descending - most critical jobs first)
    # Secondary key: best_completions (ascending - earliest completion first)
    # Tertiary key: flexibilities (ascending - fewer options first)
    
    # Create a composite score for lexicographic ordering
    # Normalize each component to [0, 1] range, then combine with large weight gaps
    
    valid_critical = job_critical[np.isfinite(job_critical)]
    if len(valid_critical) > 0:
        crit_min = np.min(valid_critical)
        crit_max = np.max(valid_critical)
        crit_range = crit_max - crit_min
    else:
        crit_min, crit_max, crit_range = 0.0, 1.0, 1.0
    
    valid_comps = best_completions[np.isfinite(best_completions)]
    if len(valid_comps) > 0:
        comp_min = np.min(valid_comps)
        comp_max = np.max(valid_comps)
        comp_range = comp_max - comp_min
    else:
        comp_min, comp_max, comp_range = 0.0, 1.0, 1.0
    
    valid_flex = flexibilities[flexibilities > 0]
    if len(valid_flex) > 0:
        flex_min = np.min(valid_flex)
        flex_max = np.max(valid_flex)
        flex_range = flex_max - flex_min
    else:
        flex_min, flex_max, flex_range = 0.0, 1.0, 1.0
    
    # Composite score: large gap between lexicographic levels
    # Level 1 (weight 10000): job critical path (higher = better)
    # Level 2 (weight 100): earliest completion (lower = better, so invert)
    # Level 3 (weight 1): flexibility (lower = better, so invert)
    
    scores = np.zeros(n)
    for i in range(n):
        if not np.isfinite(best_completions[i]):
            scores[i] = -float('inf')
            continue
        
        # Normalize to [0, 1]
        if crit_range > 1e-9:
            crit_norm = (job_critical[i] - crit_min) / crit_range
        else:
            crit_norm = 0.5
        
        if comp_range > 1e-9:
            comp_norm = 1.0 - (best_completions[i] - comp_min) / comp_range  # invert: lower comp = higher score
        else:
            comp_norm = 0.5
        
        if flex_range > 1e-9:
            flex_norm = 1.0 - (flexibilities[i] - flex_min) / flex_range  # invert: fewer options = higher score
        else:
            flex_norm = 0.5
        
        # Lexicographic weights: 10000:100:1
        scores[i] = 10000.0 * crit_norm + 100.0 * comp_norm + 1.0 * flex_norm

    # Select best operation
    best_idx = int(np.argmax(scores))
    
    # Machine selection: minimize projected makespan impact
    # Look-ahead: estimate total load each machine will receive from all ready ops
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx
    
    # Estimate future load on each machine from other ready operations
    # (excluding the one we're selecting)
    future_load = np.zeros(_MACHINE_COUNT)
    for i, op in enumerate(ready_operations):
        if i == best_idx:
            continue
        mo = op.get("machine_options", {})
        # For each machine option of other ops, add the minimum duration as a proxy
        for m, d in mo.items():
            if m < _MACHINE_COUNT:
                future_load[m] += d * 0.5  # discount by 0.5 since not all will use this machine
    
    best_machine = None
    best_score = float('inf')
    avg_load = np.mean(_machine_load)
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Projected makespan impact: completion time + future load penalty
            # Penalize machines that will be heavily loaded
            future_penalty = 0.2 * future_load[m]
            # Also penalize if this machine is already the most loaded
            load_imbalance = max(0, _machine_load[m] - avg_load)
            imbalance_penalty = 0.1 * load_imbalance
            score = completion + future_penalty + imbalance_penalty
        else:
            score = time + d
        
        if score < best_score:
            best_score = score
            best_machine = m
    
    # Update state
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
        # Update job critical path estimate
        j = ready_operations[best_idx]["job"]
        if j in _job_remaining_critical:
            _job_remaining_critical[j] = max(0, _job_remaining_critical[j] - machine_options[best_machine])
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
