"""Rewritten dispatch rule for job-shop scheduling (event-driven, fixed order)."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _soft_min(arr, temperature=1.0):
    """Smooth approximation of min via log-sum-exp (negative log-sum-exp)."""
    if arr.size == 0:
        return 0.0
    a = arr / temperature
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(np.min(arr))
    shifted = a - a_max
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (a_max + log_sum) * temperature

def _soft_max(arr, temperature=1.0):
    """Smooth approximation of max via log-sum-exp."""
    if arr.size == 0:
        return 0.0
    a = arr / temperature
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(np.max(arr))
    shifted = a - a_max
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (a_max + log_sum) * temperature

def _normalize(arr):
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with keys:
            - job: job identifier
            - op: operation identifier  
            - machine_options: dict {machine: duration} or similar
            - remaining_ops: number of remaining operations in the job
            - remaining_work: total remaining processing time for the job
        time: current simulation time
    
    Returns:
        int: index of the ready operation to start next
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=np.float64)
    
    # Compute duration stats and machine flexibility
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    mean_durs = np.zeros(n, dtype=np.float64)
    n_machines = np.zeros(n, dtype=np.float64)
    flexibility = np.zeros(n, dtype=np.float64)  # lower = more constrained
    
    for i, op in enumerate(ready_operations):
        opts = op.get("machine_options", {})
        if isinstance(opts, dict) and opts:
            vals = list(opts.values())
            if vals:
                min_durs[i] = min(vals)
                max_durs[i] = max(vals)
                mean_durs[i] = np.mean(vals)
                n_machines[i] = len(vals)
                # Flexibility: ratio of min to max duration spread; also count
                spread = max(vals) - min(vals)
                flexibility[i] = len(vals) + spread / (mean_durs[i] + 1e-9)
            else:
                min_durs[i] = 0
                max_durs[i] = 0
                mean_durs[i] = 0
                n_machines[i] = 1
                flexibility[i] = 1.0
        elif isinstance(opts, (list, tuple, np.ndarray)):
            if len(opts) > 0:
                min_durs[i] = min(opts)
                max_durs[i] = max(opts)
                mean_durs[i] = np.mean(opts)
                n_machines[i] = len(opts)
                spread = max(opts) - min(opts)
                flexibility[i] = len(opts) + spread / (mean_durs[i] + 1e-9)
            else:
                min_durs[i] = 0
                max_durs[i] = 0
                mean_durs[i] = 0
                n_machines[i] = 1
                flexibility[i] = 1.0
        else:
            min_durs[i] = 0
            max_durs[i] = 0
            mean_durs[i] = 0
            n_machines[i] = 1
            flexibility[i] = 1.0
    
    # Normalize each component to [0, 1]
    c_work = _normalize(remaining_works)
    
    # Use soft-min surrogate for duration: smoother than hard min
    # Lower soft-min duration = higher priority (invert)
    # Use temperature relative to scale of durations
    dur_scale = max(np.max(mean_durs), 1.0)
    soft_min_durs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Reconstruct per-operation durations for soft-min
        opts = ready_operations[i].get("machine_options", {})
        if isinstance(opts, dict) and opts:
            vals = np.array(list(opts.values()), dtype=np.float64)
            if vals.size > 0:
                soft_min_durs[i] = _soft_min(vals, temperature=dur_scale * 0.1)
            else:
                soft_min_durs[i] = 0
        elif isinstance(opts, (list, tuple, np.ndarray)):
            if len(opts) > 0:
                vals = np.array(opts, dtype=np.float64)
                soft_min_durs[i] = _soft_min(vals, temperature=dur_scale * 0.1)
            else:
                soft_min_durs[i] = 0
        else:
            soft_min_durs[i] = 0
    
    c_dur = 1.0 - _normalize(soft_min_durs)
    
    c_ops = _normalize(remaining_ops)
    
    # Machine flexibility: lower flexibility (fewer options, tighter spread) = higher urgency
    # Invert so that constrained machines get higher score
    c_flex = 1.0 - _normalize(flexibility)
    
    # Multiplicative urgency-constraint interaction:
    # Jobs with high remaining work AND low flexibility (constrained) get boosted
    # This reduces blocking on bottleneck machines
    urgency = c_work * c_ops  # combined urgency signal
    constraint_pressure = c_flex  # how constrained this op is
    
    # Multiplicative interaction: urgency is amplified when constraint is high
    interaction = urgency * (1.0 + 0.5 * constraint_pressure)
    
    # Additive base with multiplicative boost
    base_score = 0.45 * c_work + 0.30 * c_dur + 0.15 * c_ops + 0.10 * c_flex
    scores = base_score + 0.25 * interaction
    
    # Small random tiebreaker
    scores += _random.random() * 1e-8
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
