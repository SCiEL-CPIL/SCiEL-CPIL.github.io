"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
# We track a global start time and enforce a safety margin.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds

def _elapsed():
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # Extract key features for each ready operation
    # We'll compute a score for each operation and pick the best one.
    # The score should balance:
    # 1. Remaining work (urgency: more work = higher priority)
    # 2. Minimum processing time (shortest processing time first reduces makespan)
    # 3. Machine load balance (prefer machines with lower expected load)
    # 4. Number of remaining operations in the job (critical path consideration)
    
    best_score = float('-inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            # No machines available; assign lowest priority
            score = float('-inf')
        else:
            # Compute features
            durations = list(machine_options.values())
            min_duration = min(durations) if durations else 1.0
            avg_duration = sum(durations) / len(durations) if durations else 1.0
            
            # Use soft-min as a smooth surrogate for min duration
            # This penalizes operations where all machines are slow, not just the fastest one
            if durations:
                d_arr = np.array(durations, dtype=float)
                min_d = np.min(d_arr)
                # logsumexp(-d_arr) = -min_d + log(sum(exp(-(d_arr - min_d))))
                # soft_min = -logsumexp(-d_arr) = min_d - log(sum(exp(-(d_arr - min_d))))
                exp_terms = np.exp(-(d_arr - min_d))
                log_sum = np.log(np.sum(exp_terms))
                soft_min_duration = min_d - log_sum
            else:
                soft_min_duration = 1.0

            if min_duration <= 0:
                min_duration = 1.0
            if avg_duration <= 0:
                avg_duration = 1.0
            if soft_min_duration <= 0:
                soft_min_duration = 1.0
            
            # Feature 1: Remaining work (urgency)
            f1 = remaining_work
            
            # Feature 2: Shortest processing time (SPT) using soft-min
            f2 = 1.0 / soft_min_duration
            
            # Feature 3: Average processing time consideration
            f3 = 1.0 / avg_duration
            
            # Feature 4: Remaining ops in job
            if remaining_ops > 0:
                f4 = 1.0 / remaining_ops
            else:
                f4 = 1.0
            
            # Feature 5: Weighted Shortest Processing Time (WSPT) proxy
            f5 = f1 * f2
            
            # Weighted combination of features
            # Increased weight on WSPT (f5) and SPT (f2)
            w1 = 1.0  # remaining work
            w2 = 2.0  # SPT (soft)
            w3 = 0.5  # avg duration
            w4 = 0.5  # remaining ops
            w5 = 4.0  # work * SPT combo
            
            score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
        
        if score > best_score:
            best_score = score
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
