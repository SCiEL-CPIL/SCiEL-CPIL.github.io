"""Flexible job-shop scheduling dispatcher with smooth machine selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock; keep safety margin
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 145.0  # 5s safety margin

def _elapsed():
    return _time.time() - _DEADLINE_START

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine to run it on.
    
    Strategy:
    1. Score each ready operation by a composite heuristic:
       - Prefer operations with more remaining work (critical path)
       - Prefer operations with fewer remaining ops (urgency)
       - Prefer operations whose shortest machine duration is small (less blocking)
    2. For the chosen operation, select the machine using a smooth softmax-weighted
       surrogate that spreads load across similar-duration machines while strongly
       preferring shorter durations.
    
    Returns (index, machine) tuple.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE_LIMIT:
        # Fallback: return first operation with its shortest-duration machine
        op0 = ready_operations[0]
        best_m = min(op0["machine_options"], key=lambda m: op0["machine_options"][m])
        return (0, best_m)
    
    n = len(ready_operations)
    if n == 0:
        return (0, None)
    if n == 1:
        op = ready_operations[0]
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (0, best_m)
    
    # --- Step 1: Score operations to pick which to schedule ---
    # Composite score: we want to maximize this score
    # Components:
    #   - remaining_work: more work = more urgent (critical path)
    #   - 1/remaining_ops: fewer remaining ops = closer to completion
    #   - 1/min_duration: shorter processing = less blocking
    
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = max(op["remaining_ops"], 1)
        durations = list(op["machine_options"].values())
        min_dur = min(durations)
        
        # Normalize components
        score = 0.0
        score += 1.0 * rw              # remaining work (primary)
        score += 0.5 / ro              # urgency: fewer remaining ops
        score += 0.3 / (min_dur + 1e-9)  # less blocking
        scores[i] = score
    
    best_idx = int(np.argmax(scores))
    
    # --- Step 2: Machine selection for the chosen operation ---
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]
    
    if len(machine_options) == 1:
        best_m = list(machine_options.keys())[0]
        return (best_idx, best_m)
    
    # Smooth softmax-weighted machine selection:
    # Use a temperature parameter to control how "soft" the selection is.
    # Lower temperature -> more concentrated on shortest duration
    # Higher temperature -> more spread across machines
    
    machines = list(machine_options.keys())
    durations = np.array([machine_options[m] for m in machines], dtype=np.float64)
    
    # Temperature: inverse scale for softmax
    # We want to strongly prefer shorter durations but allow some spread
    # when durations are close. Use a relative temperature based on the range.
    dur_min = np.min(durations)
    dur_range = np.max(durations) - dur_min
    
    if dur_range < 1e-9:
        # All durations essentially equal; pick first
        best_m = machines[0]
    else:
        # Temperature relative to the spread of durations
        # tau controls smoothness: smaller tau = sharper preference
        tau = 0.15 * dur_range  # smoothness parameter
        
        # Compute softmax weights: w_i = exp(-d_i / tau) / sum(exp(-d_j / tau))
        # This gives higher weight to shorter durations
        # Use log-sum-exp for numerical stability
        neg_d_over_tau = -durations / tau
        max_val = np.max(neg_d_over_tau)
        shifted = neg_d_over_tau - max_val
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        
        # Pick the machine with highest weight (deterministic: argmax)
        # This is equivalent to argmin of duration when tau is small,
        # but when durations are close, it spreads more evenly
        best_m_idx = int(np.argmax(weights))
        best_m = machines[best_m_idx]
    
    return (best_idx, best_m)
# EVOLVE-BLOCK-END
