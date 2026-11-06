"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock.
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
    # 5. Constraint tightness: fewest machine options = higher priority
    
    best_score = float('-inf')
    best_index = 0
    best_machine = None
    
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
            machines = list(machine_options.keys())
            num_machines = len(machines)
            
            min_duration = min(durations) if durations else 1.0
            avg_duration = sum(durations) / len(durations) if durations else 1.0
            max_duration = max(durations) if durations else 1.0
            
            if min_duration <= 0:
                min_duration = 1.0
            if avg_duration <= 0:
                avg_duration = 1.0
            
            # Feature 1: Remaining work (urgency)
            # Higher remaining work should be prioritized
            f1 = remaining_work
            
            # Feature 2: Shortest processing time (SPT)
            # Operations that can be completed quickly should be prioritized
            f2 = 1.0 / min_duration
            
            # Feature 3: Average processing time consideration
            f3 = 1.0 / avg_duration
            
            # Feature 4: Remaining ops in job (jobs with fewer remaining ops are closer to completion)
            # Prioritize jobs that are closer to completion to reduce WIP
            if remaining_ops > 0:
                f4 = 1.0 / remaining_ops
            else:
                f4 = 1.0
            
            # Feature 5: Combine remaining work with min duration (throughput proxy)
            f5 = f1 * f2
            
            # Feature 6: Constraint tightness - fewer machine options means higher constraint
            # Operations with fewer options are more constrained and should be prioritized
            # to prevent future starvation
            if num_machines > 0:
                f6 = 1.0 / num_machines
            else:
                f6 = 0.0
            
            # Feature 7: Duration variance - operations with low variance across machines
            # are more predictable and may be easier to schedule
            if num_machines > 1:
                d_arr = np.array(durations, dtype=float)
                variance = float(np.var(d_arr))
                # Lower variance = more predictable = slight preference
                f7 = 1.0 / (1.0 + variance)
            else:
                f7 = 1.0
            
            # Weighted combination of features
            # These weights are tuned heuristically for FJSP
            w1 = 2.0   # remaining work
            w2 = 3.0   # SPT
            w3 = 1.0   # avg duration
            w4 = 1.5   # remaining ops
            w5 = 2.5   # work * SPT combo
            w6 = 4.0   # constraint tightness (fewer machines = higher priority)
            w7 = 0.5   # duration variance (lower is better)
            
            score = (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + 
                     w5 * f5 + w6 * f6 + w7 * f7)
        
        if score > best_score:
            best_score = score
            best_index = i
            # Select the machine with minimum duration for this operation
            if machine_options:
                best_machine = min(machine_options, key=machine_options.get)
    
    # Return just the index as per the interface
    return int(best_index)
# EVOLVE-BLOCK-END
