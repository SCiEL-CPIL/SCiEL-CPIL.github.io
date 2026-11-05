"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: track start time to ensure we finish well before the 150s limit
_DISPATCH_START = None

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Uses a refined composite heuristic with smooth surrogate for min/max durations.
    """
    global _DISPATCH_START
    if _DISPATCH_START is None:
        _DISPATCH_START = _time.time()
    
    # Safety margin: if we're too close to the deadline, return a quick heuristic
    elapsed = _time.time() - _DISPATCH_START
    if elapsed > 140:  # 10s safety margin before 150s limit
        if not ready_operations:
            return 0
        return 0
    
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        work = op["remaining_work"]
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        
        if not durations:
            scores[i] = work
            continue
        
        # Use smooth surrogate (soft-min via log-sum-exp) for min duration
        # This avoids the nonsmoothness of exact min
        d_arr = np.array(durations, dtype=np.float64)
        
        # Soft-min: -log(sum(exp(-d/t))) * t, for small t
        t_soft = 0.5
        soft_min = -t_soft * np.log(np.sum(np.exp(-d_arr / t_soft)))
        soft_max = t_soft * np.log(np.sum(np.exp(d_arr / t_soft)))
        
        # Average duration
        avg_d = np.mean(d_arr)
        
        # Most work remaining - primary criterion
        score = work
        
        # Prefer operations with shorter effective minimum duration (faster completion)
        # Use soft-min instead of hard min for smoothness
        score -= 0.1 * soft_min
        
        # Consider the number of remaining operations
        remaining_ops = op.get("remaining_ops", 0)
        if remaining_ops > 0:
            score += 0.05 * remaining_ops
        
        # Penalize operations that have very long effective maximum duration
        score -= 0.02 * soft_max
        
        # Additional: prefer operations where the best machine is less contested
        # (heuristic: shorter durations on any machine are better for throughput)
        score -= 0.01 * avg_d
        
        scores[i] = score
    
    # Add a small random tiebreaker to avoid systematic bias
    # Use a deterministic seed based on time to be reproducible
    seed = int(time * 1000) % 100000
    rng = np.random.RandomState(seed)
    tiebreaker = rng.uniform(0, 0.01, size=n)
    scores += tiebreaker
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
