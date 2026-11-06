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
            # Machine selection: use soft-min to prefer machines that minimize
            # a combination of immediate duration and expected blocking
            best_machine_for_op = None
            min_duration = float('inf')
            
            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
            
            # For each machine, compute a cost that balances:
            # - immediate processing time (dur)
            # - expected idle time for subsequent ops on this job
            # Use a smooth approximation: dur * (1 + alpha * remaining_ops * dur / (norm_rw + 1))
            alpha = 0.4
            best_machine_cost = float('inf')
            for m, dur in machine_options.items():
                # Cost: immediate work + expected delay due to being on a busy machine
                # The term dur^2 / (norm_rw+1) approximates the quadratic cost of
                # long operations blocking the job's critical path
                cost = dur * (1.0 + alpha * remaining_ops * dur / (norm_rw + 1.0))
                if cost < best_machine_cost:
                    best_machine_cost = cost
                    best_machine_for_op = m
            
            selected_duration = machine_options[best_machine_for_op]
            
            # Score components using a ratio-based approach (WSJF-like):
            # Urgency / Cost where:
            # - Urgency: remaining_work / remaining_ops (work per remaining op)
            # - Cost: selected_duration (time to complete this op)
            
            # 1. Work-per-op ratio: higher means more urgent
            work_per_op = remaining_work / (remaining_ops + 1e-9)
            
            # 2. Normalize work_per_op for scaling
            max_wpo = 0
            for op2 in ready_operations:
                rw2 = op2.get("remaining_work", 0)
                ro2 = op2.get("remaining_ops", 0)
                wpo2 = rw2 / (ro2 + 1e-9)
                if wpo2 > max_wpo:
                    max_wpo = wpo2
            norm_wpo = max_wpo if max_wpo > 0 else 1.0
            
            normalized_wpo = work_per_op / norm_wpo
            
            # 3. Cost component: shorter duration is better
            #    Use reciprocal to make it a "score" (higher is better)
            cost_score = 1.0 / (selected_duration + 1e-9)
            
            # 4. Normalize cost across all ready ops
            max_cost_score = 0
            for op2 in ready_operations:
                mo2 = op2.get("machine_options", {})
                if mo2:
                    min_dur2 = min(mo2.values())
                    cs2 = 1.0 / (min_dur2 + 1e-9)
                    if cs2 > max_cost_score:
                        max_cost_score = cs2
            norm_cost = max_cost_score if max_cost_score > 0 else 1.0
            normalized_cost = cost_score / norm_cost
            
            # 5. Remaining ops penalty: fewer remaining ops = closer to completion
            #    This captures the idea that finishing a job frees up resources
            urgency_bonus = 1.0 / (1.0 + 0.3 * remaining_ops)
            
            # 6. Deterministic tiebreaker
            job = op.get("job", 0)
            op_num = op.get("op", 0)
            tie_break = ((job * 31 + op_num * 17) % 97) * 0.00005
            
            # Weighted combination:
            # - normalized_wpo: primary driver (which job has most work left per op)
            # - normalized_cost: secondary (prefer faster completion)
            # - urgency_bonus: tertiary (prefer jobs closer to done)
            score = (
                0.55 * normalized_wpo +
                0.30 * normalized_cost +
                0.15 * urgency_bonus +
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
