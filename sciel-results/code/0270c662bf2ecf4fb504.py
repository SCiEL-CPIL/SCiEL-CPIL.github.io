"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for the simulator (per evaluation run)
_sim_state = {}
_DEADLINE = 115.0  # Safety margin before 120s limit
_start_time = None
_best_score = None
_best_dispatch_fn = None

def _reset_sim_state():
    global _sim_state, _start_time, _best_score, _best_dispatch_fn
    _sim_state = {}
    _start_time = time.time()
    _best_score = float('inf')
    _best_dispatch_fn = None

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule that selects which ready operation to start next.
    
    Uses a refined heuristic combining:
    - MWR (Most Work Remaining) for job selection
    - SPT (Shortest Processing Time) for machine selection
    - Adaptive weighting based on simulation progress
    - Machine load balancing considerations
    """
    # Initialize global state on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return 0
    
    # Time guard: if we're running out of time, use a fast simple rule
    if _check_time():
        return max(range(len(ready_operations)), 
                  key=lambda i: ready_operations[i]["remaining_work"])
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Track simulation progress for adaptive weighting
    total_remaining_work = sum(op["remaining_work"] for op in ready_operations)
    total_remaining_ops = sum(op["remaining_ops"] for op in ready_operations)
    
    # Estimate progress: fraction of work completed
    # We don't have initial work, so use relative measures
    progress_estimate = time / (time + 100)  # Simple progress estimate
    
    # Adaptive weights based on progress
    # Early: prioritize MWR to start long jobs
    # Late: prioritize SPT to clear short jobs quickly
    w_work = 1.0 - 0.3 * progress_estimate
    w_ops = 0.2 + 0.3 * progress_estimate
    w_spt = 0.1 + 0.5 * progress_estimate
    
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations)
            max_duration = max(durations)
            avg_duration = sum(durations) / len(durations)
            # Flexibility: how many machines can do this op
            flexibility = len(machine_options)
        else:
            min_duration = 0
            max_duration = 0
            avg_duration = 0
            flexibility = 0
        
        # MWR component: prioritize jobs with most remaining work
        mwr_score = remaining_work
        
        # Operations remaining component
        ops_score = remaining_ops * 10  # Scale to comparable range
        
        # SPT component: prefer operations with shorter processing times
        # Normalize by max duration to make comparable
        spt_score = (1.0 / (min_duration + 1e-9)) if min_duration > 0 else 100.0
        
        # Flexibility component: prefer ops with more machine options
        flex_score = flexibility
        
        # Combine scores with adaptive weights
        score = (w_work * mwr_score + 
                 w_ops * ops_score + 
                 w_spt * spt_score +
                 0.05 * flex_score)
        
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = max(range(n), key=lambda i: scores[i])
    return int(best_idx)

# EVOLVE-BLOCK-END
