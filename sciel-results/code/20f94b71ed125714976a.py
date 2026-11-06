"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Track machine load across dispatch calls to inform machine selection
_machine_load = {}
_last_time = None

def _update_machine_load(ready_operations, time):
    """Update machine load estimates based on time progression."""
    global _last_time
    if _last_time is not None and time > _last_time:
        dt = time - _last_time
        for m in _machine_load:
            _machine_load[m] -= dt
            if _machine_load[m] < 0:
                _machine_load[m] = 0.0
    _last_time = time

def dispatch(ready_operations, time):
    """Dispatch rule: prioritize by workload-weighted tightness, select shortest machine."""
    _update_machine_load(ready_operations, time)
    
    n = len(ready_operations)
    if n == 0:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        # Find the best machine for this operation
        machine_opts = op["machine_options"]
        min_dur = float('inf')
        best_m_for_op = None
        
        for m, dur in machine_opts.items():
            if dur < min_dur:
                min_dur = dur
                best_m_for_op = m
        
        # Tightness: remaining work / minimum processing time (higher = more urgent)
        remaining_work = op.get("remaining_work", 0)
        if min_dur > 0:
            tightness = remaining_work / min_dur
        else:
            tightness = 0.0
        
        # Machine load penalty: prefer less loaded machines
        load_penalty = _machine_load.get(best_m_for_op, 0.0)
        
        # Combined score: prioritize high tightness, penalize loaded machines
        # Use a weighted combination
        score = tightness - 0.1 * load_penalty
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
    
    # Update machine load for the selected operation
    if best_machine is not None:
        machine_opts = ready_operations[best_idx]["machine_options"]
        dur = machine_opts.get(best_machine, 0)
        _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + dur
    
    return best_idx
# EVOLVE-BLOCK-END
