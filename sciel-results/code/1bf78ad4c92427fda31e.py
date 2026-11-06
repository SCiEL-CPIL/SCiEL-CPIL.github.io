"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math


def _soft_min(values, beta=0.5):
    """
    Soft-min via log-sum-exp for smooth surrogate of min().
    As beta -> 0, approaches true min. Larger beta = smoother.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Use log-sum-exp trick for numerical stability
    min_val = min(values)
    # Shift to avoid overflow
    shifted = [v - min_val for v in values]
    
    # Soft-min: -1/beta * log(sum(exp(-beta * v)))
    # Equivalent to: min_val - 1/beta * log(sum(exp(-beta * (v - min_val))))
    exp_sum = 0.0
    for s in shifted:
        exp_sum += math.exp(-beta * s)
    
    if exp_sum <= 0:
        return min_val
    
    return min_val - (1.0 / beta) * math.log(exp_sum)


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Combine multiple heuristics with soft-min for machine duration:
    1. Critical Path: Prioritize operations with longest remaining work
    2. Soft-min duration: Use log-sum-exp to capture distribution of machine speeds
    3. Urgency: Remaining ops count as urgency signal
    4. Machine contention: Prefer operations with less contended machine options
    
    Weights tuned for job-shop makespan minimization.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0
    
    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 3.0  # 3 second safety margin
    
    # Precompute normalization constants
    max_remaining_work = 0
    max_remaining_ops = 0
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        max_remaining_work = max(max_remaining_work, rw)
        max_remaining_ops = max(max_remaining_ops, ro)
    
    # Compute scores
    scores = [0.0] * n
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            # Soft-min captures the "effective" shortest duration considering alternatives
            soft_min_dur = _soft_min(durations, beta=0.3)
            min_dur = min(durations)
            num_machines = len(durations)
        else:
            soft_min_dur = 1.0
            min_dur = 1.0
            num_machines = 1
        
        # Work score: normalize remaining work
        if max_remaining_work > 0:
            work_score = remaining_work / max_remaining_work
        else:
            work_score = 0.0
        
        # Urgency: jobs with fewer remaining ops are closer to completion
        # Invert: fewer remaining ops = higher urgency (should finish soon)
        if max_remaining_ops > 0:
            urgency_score = 1.0 - (remaining_ops / max_remaining_ops)
        else:
            urgency_score = 0.0
        
        # Duration score: shorter effective duration is better
        # Use soft-min for smoother gradient
        if soft_min_dur > 0:
            duration_score = 1.0 / (soft_min_dur + 1.0)
        else:
            duration_score = 1.0
        
        # Machine flexibility: more machine options = more flexible
        flexibility_score = min(num_machines, 5) / 5.0
        
        # Weighted combination
        # Emphasize work (critical path) and duration, with urgency and flexibility
        scores[i] = (
            0.45 * work_score +
            0.30 * duration_score +
            0.15 * urgency_score +
            0.10 * flexibility_score
        )
    
    # Check deadline
    if _time.time() > deadline - 0.5:
        best_idx = 0
        for i in range(1, n):
            if scores[i] > scores[best_idx]:
                best_idx = i
        return int(best_idx)
    
    # Second pass: machine contention adjustment
    # Count how many ready operations can use each machine
    machine_contention = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine in machine_options:
            machine_contention[machine] = machine_contention.get(machine, 0) + 1
    
    max_contention = max(machine_contention.values()) if machine_contention else 1
    
    # Adjust scores: prefer operations that can use less contended machines
    adjusted_scores = [0.0] * n
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            adjusted_scores[i] = scores[i]
            continue
        
        # Find minimum contention among available machines
        min_contention = float('inf')
        for machine in machine_options:
            contention = machine_contention.get(machine, 0)
            if contention < min_contention:
                min_contention = contention
        
        # Contention score: lower contention = higher score
        if max_contention > 0:
            contention_score = 1.0 - (min_contention / max_contention)
        else:
            contention_score = 0.0
        
        adjusted_scores[i] = scores[i] * 0.85 + contention_score * 0.15
    
    # Select the operation with highest score
    best_idx = 0
    for i in range(1, n):
        if adjusted_scores[i] > adjusted_scores[best_idx]:
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
