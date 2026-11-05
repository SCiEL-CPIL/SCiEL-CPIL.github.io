"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

# Flow-network inspired state:
# Track per-job remaining work to estimate "distance to sink"
_job_remaining_work = {}  # job_id -> remaining work
_job_total_work = {}      # job_id -> total work (first seen)
_machine_capacity_used = np.zeros(_MACHINE_COUNT)  # cumulative processing assigned

def _reset_state():
    global _machine_load, _job_remaining_work, _job_total_work, _machine_capacity_used
    _machine_load = np.zeros(_MACHINE_COUNT)
    _job_remaining_work = {}
    _job_total_work = {}
    _machine_capacity_used = np.zeros(_MACHINE_COUNT)

def dispatch(ready_operations, time):
    """
    Dispatch rule using flow-network bottleneck capacity model.
    Each operation's priority = bottleneck capacity along the path
    from job source -> operation -> machine sink, weighted by downstream unlock.
    """
    global _machine_load, _last_seen_time, _job_remaining_work, _job_total_work, _machine_capacity_used

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance
    if time < _last_seen_time:
        _reset_state()
    _last_seen_time = time

    # Update job tracking
    for op in ready_operations:
        j = op["job"]
        if j not in _job_total_work:
            _job_total_work[j] = op["remaining_work"]
        _job_remaining_work[j] = min(_job_remaining_work.get(j, op["remaining_work"]), op["remaining_work"])

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
                _machine_capacity_used[best_machine] += machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Flow-network model:
    # For each ready operation, compute:
    # 1. Source capacity: how much work remains in the job (higher = more critical)
    # 2. Edge capacity: earliest completion time on best machine (lower = better)
    # 3. Sink capacity: machine's residual capacity (lower load = better)
    # 4. Bottleneck: min of normalized capacities along the path
    
    scores = np.zeros(n)
    earliest_completions = np.zeros(n)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        j = op["job"]
        rw = op["remaining_work"]
        mo = op["machine_options"]
        
        if not mo:
            scores[i] = -float('inf')
            earliest_completions[i] = float('inf')
            continue
        
        # Source capacity: remaining work in job (normalized)
        total_j = _job_total_work.get(j, rw)
        source_cap = rw / max(total_j, 1.0) if total_j > 0 else 0.0
        
        # Edge capacity: earliest completion time
        best_comp = float('inf')
        best_m = None
        for m, d in mo.items():
            if m < _MACHINE_COUNT:
                start = max(time, _machine_load[m])
                comp = start + d
            else:
                comp = time + d
            if comp < best_comp:
                best_comp = comp
                best_m = m
        
        earliest_completions[i] = best_comp
        best_machines[i] = best_m
        
        # Sink capacity: machine's current load relative to average
        if best_m is not None and best_m < _MACHINE_COUNT:
            avg_load = np.mean(_machine_load)
            sink_cap = 1.0 - min(1.0, max(0.0, (_machine_load[best_m] - avg_load) / max(avg_load, 1.0)))
        else:
            sink_cap = 0.5
        
        # Downstream unlock: remaining work after this op
        remaining_after = max(0, rw - min(d for d in mo.values()))
        unlock_val = remaining_after / max(total_j, 1.0)
        
        # Flexibility: fewer options = more constrained = higher priority
        flex = 1.0 / len(mo)
        
        # Composite score: bottleneck capacity model
        # Priority = min(source, edge_inv, sink) * unlock * flex
        # Edge inv: lower completion = higher value
        max_ec = np.max(earliest_completions[:i+1]) if i > 0 else best_comp
        min_ec = np.min(earliest_completions[:i+1]) if i > 0 else best_comp
        if max_ec - min_ec > 1e-9:
            edge_inv = 1.0 - (best_comp - min_ec) / (max_ec - min_ec)
        else:
            edge_inv = 0.5
        
        # Bottleneck: geometric mean of capacities (smooth min)
        # Use product of normalized values as proxy for bottleneck
        bottleneck = (source_cap + 1e-6) * (edge_inv + 1e-6) * (sink_cap + 1e-6)
        
        # Weighted combination emphasizing bottleneck and unlock
        scores[i] = 1.0 * bottleneck + 0.7 * unlock_val + 0.5 * flex

    # Select best operation
    best_idx = int(np.argmax(scores))
    
    # Machine selection: minimize completion with load-balancing penalty
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx
    
    best_machine = None
    best_score = float('inf')
    avg_load = np.mean(_machine_load)
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Penalty for assigning to already-loaded machines
            load_penalty = 0.2 * max(0, _machine_load[m] - avg_load)
            score = completion + load_penalty
        else:
            score = time + d
        
        if score < best_score:
            best_score = score
            best_machine = m
    
    # Update state
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
        _machine_capacity_used[best_machine] += machine_options[best_machine]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
