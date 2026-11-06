"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We keep a global start time and a per-call budget.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 2.0  # seconds before the 150s limit
_CALL_BUDGET = 0.05     # max seconds per dispatch call (50ms)


def _elapsed():
    return _time.time() - _START_TIME


def _over_budget():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and, optionally, which machine.
    
    Strategy:
    1. Constraint-tightness: prefer operations with fewer machine options (scarce resources).
    2. Among those, prefer operations with more remaining work (longest remaining processing time).
    3. For machine selection, pick the machine with the minimum duration (to reduce idle gaps).
    
    Returns: (index, machine) if machine selection is beneficial, else just index.
    """
    if _over_budget():
        # Fallback: just pick the first ready operation
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # If there are multiple machine options, pick the one with min duration
        mo = op.get("machine_options", {})
        if mo and len(mo) > 1:
            best_machine = min(mo, key=lambda m: mo[m])
            return (0, best_machine)
        return 0

    # Compute scores for each ready operation
    # Primary: constraint tightness (fewer machine options = higher priority)
    # Secondary: remaining work (more work = higher priority)
    # Tertiary: remaining operations (more ops remaining = higher priority)
    
    scores = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        n_machines = len(mo) if mo else 1
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Constraint tightness: fewer options -> higher score
        # We invert: 1/n_machines so that fewer options gives higher value
        # Boost tightness significantly to prevent starvation
        tightness = 1.0 / max(n_machines, 1)
        
        # Remaining work: normalize roughly
        work_score = float(remaining_work)
        
        # Remaining operations
        ops_score = float(remaining_ops)
        
        # Combined score: tightness dominates, then work, then ops
        # Use a weighted sum with higher weight on tightness
        score = 20.0 * tightness + work_score + 0.05 * ops_score
        scores[i] = score

    best_idx = int(np.argmax(scores))
    
    # Machine selection: for the chosen operation, pick the machine that minimizes
    # projected completion time, considering current time and duration
    op = ready_operations[best_idx]
    mo = op.get("machine_options", {})
    if mo and len(mo) > 1:
        # Pick machine with minimum duration (to reduce makespan contribution)
        best_machine = min(mo, key=lambda m: mo[m])
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
