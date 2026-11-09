"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard
_START_TIME = time_module.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time_module.time() - _START_TIME) < (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Check deadline
    if not _elapsed_ok():
        # Fallback to simple index to avoid timeout
        return 0

    # Extract durations. 
    # machine_options is {machine: duration}. 
    # We assume the operation is assigned to one of these. 
    # If multiple, we might take the max or min? 
    # In open shop, usually one machine per op. 
    # Let's take the max duration available as a proxy for "heaviness" if flexible, 
    # or just the single value if fixed.
    
    # Extract features for each ready operation
    durations = []
    remaining_works = []
    remaining_ops_list = []
    flexibilities = []
    
    for op in ready_operations:
        options = op.get("machine_options", {})
        if options:
            # Use max duration as conservative estimate of delay
            duration = max(options.values())
            flexibility = len(options)
        else:
            duration = 0
            flexibility = 0
            
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        durations.append(duration)
        remaining_works.append(remaining_work)
        remaining_ops_list.append(remaining_ops)
        flexibilities.append(flexibility)
    
    if not durations:
        return 0
    
    # Convert to numpy arrays for vectorized computation
    durations = np.array(durations, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops = np.array(remaining_ops_list, dtype=np.float64)
    flexibilities = np.array(flexibilities, dtype=np.float64)
    
    # Normalize features to [0, 1] range for weighted combination
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min < 1e-9:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    norm_duration = normalize(durations)
    norm_remaining_work = normalize(remaining_works)
    norm_flexibility = normalize(flexibilities)
    
    # Work density: remaining work per remaining operation
    # Avoid division by zero
    safe_remaining_ops = np.where(remaining_ops > 0, remaining_ops, 1.0)
    work_density = remaining_works / safe_remaining_ops
    norm_work_density = normalize(work_density)
    
    # Composite score: weighted combination of heuristics
    # LPT is primary, MRW and work density are secondary
    # Weights tuned for open-shop makespan minimization
    w_duration = 0.5
    w_remaining_work = 0.25
    w_work_density = 0.15
    w_flexibility = 0.10  # Slight preference for less flexible ops (earlier scheduling)
    
    # For flexibility, we want to prioritize LESS flexible ops (fewer machine options)
    # because they have fewer alternatives, so use (1 - norm_flexibility)
    score = (w_duration * norm_duration + 
             w_remaining_work * norm_remaining_work + 
             w_work_density * norm_work_density + 
             w_flexibility * (1.0 - norm_flexibility))
    
    # Fallback: if all scores are identical (e.g., all zeros), use duration then remaining_work
    if np.allclose(score, score[0]):
        if np.allclose(durations, durations[0]):
            if np.allclose(remaining_works, remaining_works[0]):
                return 0
            return int(np.argmax(remaining_works))
        return int(np.argmax(durations))
    
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
