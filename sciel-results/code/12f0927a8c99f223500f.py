"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_START = None
_DEADLINE_LIMIT = 145.0  # seconds wall-clock safety margin (evaluator kills at 150s)


def _init_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.monotonic()


def _elapsed():
    _init_deadline()
    return _time.monotonic() - _DEADLINE_START


def _remaining_time():
    return _DEADLINE_LIMIT - _elapsed()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy: Prioritize operations with highest remaining work (critical path),
    with tie-breaking by shortest processing time. Returns (index, machine)
    to guide machine selection for earliest completion.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Deadline check: if we're running low on time, pick quickly
    if _remaining_time() < 2.0:
        # Fast path: just pick the op with highest remaining work
        best_i = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_i = i
        mo = ready_operations[best_i].get("machine_options", {})
        if mo:
            bm = min(mo, key=mo.get)
            return (best_i, bm)
        return best_i
    
    # If only one option, return it with its best machine
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0
    
    best_score = -1e18
    best_idx = 0
    best_machine = None
    
    # Find max remaining work for normalization
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    
    norm_rw = max_rw if max_rw > 0 else 1.0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            score = 0
            best_machine_for_op = None
        else:
            # Improved machine selection: balance duration with future work
            # Prefer machines that minimize total time (duration + fraction of remaining work
            # that could be blocked), using a soft approach
            best_machine_for_op = None
            best_machine_score = -1e18
            min_duration = float('inf')
            
            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
            
            for m, dur in machine_options.items():
                # Machine score: prefer shorter duration, but also consider
                # that shorter duration frees the machine sooner for next op
                # Use a smoothed penalty: dur * (1 + 0.3 * remaining_ops * dur / (norm_rw + 1))
                future_factor = 1.0 + 0.3 * remaining_ops * (dur / (norm_rw + 1.0))
                m_score = -dur * future_factor
                if m_score > best_machine_score:
                    best_machine_score = m_score
                    best_machine_for_op = m
            
            selected_duration = machine_options[best_machine_for_op]
            
            # Score components:
            # 1. Critical path: remaining work (normalized) - dominant factor
            cp_score = remaining_work / norm_rw
            
            # 2. Urgency: operations with fewer remaining ops are closer to done
            #    Use a smoother formulation
            urgency = 1.0 / (1.0 + 0.5 * remaining_ops)
            
            # 3. Shorter processing time reduces blocking - use soft-min proxy
            #    The ratio of selected duration to the best available duration
            time_ratio = selected_duration / (min_duration + 1e-9)
            time_bonus = 1.0 / (1.0 + 0.5 * (time_ratio - 1.0) * remaining_ops)
            
            # 4. Deterministic tiebreaker
            job = op.get("job", 0)
            op_num = op.get("op", 0)
            tie_break = ((job * 31 + op_num * 17) % 97) * 0.0001
            
            # Weighted combination - critical path dominates
            score = (
                0.72 * cp_score +
                0.14 * urgency +
                0.12 * time_bonus +
                tie_break
            )
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # Return (index, machine) to guide machine selection
    if best_machine is not None:
        return (int(best_idx), best_machine)
    return int(best_idx)
# EVOLVE-BLOCK-END
