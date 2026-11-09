"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random as _random
from collections import defaultdict

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit

# Track cumulative machine loads to inform load-balancing decisions
_machine_loads = defaultdict(float)
_last_time = 0


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _update_machine_loads(ready_operations, current_time):
    """
    Update our estimate of machine loads based on elapsed time.
    This is a heuristic approximation since we don't have full state.
    """
    global _last_time
    if current_time > _last_time:
        dt = current_time - _last_time
        # Heuristic: assume machines are busy proportionally to time
        # This is a rough estimate to track relative loads
        for m in _machine_loads:
            _machine_loads[m] += dt * 0.5  # Assume ~50% utilization
        _last_time = current_time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic combining load-balancing, urgency, and flexibility.
    
    Key improvements over baseline:
    1. Track approximate machine loads to avoid overloading specific machines
    2. For each operation, compute the "effective completion time" if assigned to 
       its best (least-loaded + shortest-duration) machine
    3. Prioritize operations that would reduce the makespan bottleneck
    """
    global _machine_loads, _last_time
    
    if not ready_operations:
        return 0
    
    # Deadline check - if we're close to the limit, use a simple fast heuristic
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Update machine load estimates
    _update_machine_loads(ready_operations, time)
    
    n = len(ready_operations)
    
    # First pass: compute machine-specific scores for each operation
    # For each op, find the machine that minimizes (load + duration)
    op_machine_scores = []  # (op_index, best_machine, score_on_that_machine)
    op_metrics = []  # Store metrics for each op
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            op_machine_scores.append((i, None, 0))
            op_metrics.append({
                'remaining_work': remaining_work,
                'remaining_ops': remaining_ops,
                'min_duration': 0,
                'max_duration': 0,
                'num_options': 0,
                'work_per_op': 0,
                'best_completion': 0,
                'best_machine': None
            })
            continue
        
        min_duration = min(machine_options.values())
        max_duration = max(machine_options.values())
        num_options = len(machine_options)
        work_per_op = remaining_work / max(remaining_ops, 1)
        
        # For each machine option, compute estimated completion time
        # = current estimated load on that machine + processing duration
        best_completion = float('inf')
        best_machine = None
        
        for machine, duration in machine_options.items():
            # Get current estimated load for this machine
            current_load = _machine_loads.get(machine, 0.0)
            estimated_completion = current_load + duration
            
            if estimated_completion < best_completion:
                best_completion = estimated_completion
                best_machine = machine
        
        op_machine_scores.append((i, best_machine, best_completion))
        op_metrics.append({
            'remaining_work': remaining_work,
            'remaining_ops': remaining_ops,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'num_options': num_options,
            'work_per_op': work_per_op,
            'best_completion': best_completion,
            'best_machine': best_machine
        })
    
    # Second pass: compute composite score for each operation
    # We want to prioritize:
    # 1. Operations with high remaining work (they're critical to makespan)
    # 2. Operations that would complete early on their best machine (load balancing)
    # 3. Operations with low flexibility (constrained, should go first)
    # 4. Operations with high work density (urgent)
    
    # Normalize metrics for better combination
    all_remaining_work = [m['remaining_work'] for m in op_metrics]
    all_best_completion = [m['best_completion'] for m in op_metrics if m['best_completion'] > 0]
    all_work_per_op = [m['work_per_op'] for m in op_metrics]
    
    max_rw = max(all_remaining_work) if all_remaining_work else 1
    min_bc = min(all_best_completion) if all_best_completion else 1
    max_bc = max(all_best_completion) if all_best_completion else 1
    max_wpo = max(all_work_per_op) if all_work_per_op else 1
    
    scores = []
    for i, (op_idx, best_machine, best_completion) in enumerate(op_machine_scores):
        m = op_metrics[i]
        
        if m['best_machine'] is None:
            scores.append(0)
            continue
        
        # Factor 1: Remaining work (normalized) - higher is better
        f1 = m['remaining_work'] / max(max_rw, 1e-9)
        
        # Factor 2: Best completion time (normalized) - lower is better
        # Use inverse: operations that complete sooner get higher score
        if max_bc > min_bc:
            f2 = 1.0 - (m['best_completion'] - min_bc) / (max_bc - min_bc + 1e-9)
        else:
            f2 = 0.5
        
        # Factor 3: Work density (normalized) - higher is better
        f3 = m['work_per_op'] / max(max_wpo, 1e-9)
        
        # Factor 4: Flexibility - fewer options = more constrained = higher priority
        f4 = 1.0 / max(m['num_options'], 1)
        
        # Factor 5: Min duration relative to max (tightness)
        f5 = m['min_duration'] / max(m['max_duration'], 1e-9)
        
        # Factor 6: Load balance incentive - if this op's best machine is 
        # currently less loaded than average, give it a bonus
        if best_machine in _machine_loads and len(_machine_loads) > 0:
            avg_load = sum(_machine_loads.values()) / len(_machine_loads)
            load_ratio = _machine_loads[best_machine] / max(avg_load, 1e-9)
            # If machine is below average load, bonus; if above, penalty
            f6 = 1.0 / (1.0 + load_ratio)
        else:
            f6 = 0.5
        
        # Weighted combination tuned for open-shop makespan minimization
        # Emphasis on: remaining work, load balancing, urgency
        score = (
            2.0 * f1 +          # Remaining work (most important)
            1.5 * f2 +           # Load balancing (completion time on best machine)
            0.8 * f3 +           # Work density
            0.5 * f4 +           # Constrained machines first
            0.3 * f5 +           # Tightness
            1.0 * f6             # Load balance incentive
        )
        
        scores.append(score)
    
    scores = np.array(scores)
    
    # Add small random tie-breaker to avoid deterministic bias
    if n > 1:
        tie_breaker = _random.random() * 0.01
        scores = scores + tie_breaker * np.ones(n)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
