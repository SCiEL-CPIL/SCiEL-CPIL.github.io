"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import defaultdict

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0
_SOFT_MARGIN = 5.0

# Track estimated machine loads across dispatch calls
_machine_loads = defaultdict(float)
_dispatch_count = 0

def _check_deadline():
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next, and optionally which machine.
    
    Strategy: Balance machine loads while prioritizing jobs with high remaining work.
    For each candidate operation, evaluate assigning it to each available machine.
    Score based on:
    1. Resulting machine load balance (minimize max machine load)
    2. Job remaining work (higher = more urgent)
    3. Operation duration (shorter = less blocking)
    """
    global _machine_loads, _dispatch_count
    _dispatch_count += 1
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        if op["machine_options"]:
            best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
            _machine_loads[best_machine] += op["machine_options"][best_machine]
            return (0, best_machine)
        return 0
    
    if _check_deadline():
        # Fallback: pick max remaining work, assign to min-duration machine
        best_idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
        op = ready_operations[best_idx]
        if op["machine_options"]:
            best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
            _machine_loads[best_machine] += op["machine_options"][best_machine]
            return (best_idx, best_machine)
        return best_idx
    
    # For each operation, find the best machine assignment that minimizes load imbalance
    scores = np.zeros(n)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores[i] = -1e9
            continue
        
        # For each candidate machine, compute the resulting load balance score
        best_score = -1e9
        best_machine = None
        best_duration = float('inf')
        
        for machine, duration in machine_options.items():
            current_load = _machine_loads.get(machine, 0.0)
            new_load = current_load + duration
            
            # Score: we want to minimize the resulting machine load
            # Also consider the job's remaining work as urgency
            # Lower new_load is better (balances machines)
            
            # Composite: penalize high resulting load, reward high remaining work
            urgency = op["remaining_work"]
            
            # Normalize urgency to [0,1] range across all ops
            # But we don't know max remaining work without computing it first
            # Use a simple penalty: -new_load + urgency_factor * urgency
            
            # The key insight: in open-shop, makespan >= max(machine_total_work)
            # So we want to keep machine loads balanced.
            # Score = -new_load (lower is better for balance)
            # But we also want to process urgent jobs (high remaining work)
            
            # Use: score = urgency - alpha * new_load
            # Where alpha controls the tradeoff
            alpha = 0.5
            candidate_score = urgency - alpha * new_load
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_machine = machine
                best_duration = duration
        
        scores[i] = best_score
        best_machines[i] = best_machine
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Update machine load
    if best_machines[best_idx] is not None:
        op = ready_operations[best_idx]
        machine = best_machines[best_idx]
        duration = op["machine_options"][machine]
        _machine_loads[machine] += duration
        return (best_idx, machine)
    
    return best_idx
# EVOLVE-BLOCK-END
