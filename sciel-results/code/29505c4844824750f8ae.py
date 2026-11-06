import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Multi-factor heuristic combining:
    - Remaining work (MWR) to prioritize critical jobs
    - Remaining ops count to balance job completion
    - Minimum machine duration to keep machines utilized
    - Machine availability consideration (prefer ops on less busy machines)
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    start_t = time_module.perf_counter()
    DEADLINE = 145.0

    # Precompute features for all ready operations
    # We'll compute a composite score that balances multiple factors
    
    scores = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if not mo:
            scores.append(0.0)
            continue
        
        durations = list(mo.values())
        min_dur = min(durations) if durations else 1.0
        max_dur = max(durations) if durations else 1.0
        avg_dur = sum(durations) / len(durations)
        n_machines = len(mo)
        
        # Factor 1: Remaining work (higher = more urgent)
        # Normalize by total remaining work across all ops
        # We'll normalize later, so use raw for now
        
        # Factor 2: Remaining ops count (fewer ops left = closer to completion)
        # Jobs with fewer remaining ops should be completed soon
        ops_factor = remaining_ops if remaining_ops > 0 else 1
        
        # Factor 3: Duration consideration
        # Shorter operations can be completed quickly, freeing machines
        # But we also want to make progress on long operations
        # Use a balanced approach: prefer moderate durations
        dur_factor = min_dur / max(avg_dur, 1e-9)  # Ratio of min to avg
        
        # Factor 4: Machine flexibility
        # Operations with more machine options are more flexible
        flex_factor = n_machines
        
        # Composite score:
        # Primary: remaining work (most important for makespan)
        # Secondary: fewer remaining ops (complete jobs)
        # Tertiary: shorter min duration (keep flow)
        # Quaternary: more flexibility (avoid bottlenecks)
        
        # Weighted combination
        # MWR dominates, but we add small corrections
        score = rw
        
        # Add bonus for jobs with fewer remaining ops (closer to completion)
        # This helps complete jobs, which can free up their remaining resources
        score += 0.1 / ops_factor
        
        # Add small bonus for shorter operations (quick wins)
        score += 0.05 * (1.0 / max(min_dur, 1e-9))
        
        # Add small bonus for more flexible operations
        score += 0.01 * flex_factor
        
        scores.append(score)
    
    # Check deadline
    if time_module.perf_counter() - start_t > DEADLINE:
        # Return the operation with max remaining work as fallback
        best_rw = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return best_idx
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking: if scores are very close, use secondary criteria
    max_score = scores[best_idx]
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-6]
    
    if len(ties) > 1:
        # Secondary tie-breakers:
        # 1. Prefer operation with fewer remaining ops (closer to completion)
        # 2. Then prefer shorter min duration
        # 3. Then prefer more remaining work (original)
        
        def tiebreak_score(idx):
            op = ready_operations[idx]
            rw = op.get("remaining_work", 0)
            remaining_ops = op.get("remaining_ops", 0)
            mo = op.get("machine_options", {})
            durations = list(mo.values()) if mo else [1.0]
            min_dur = min(durations) if durations else 1.0
            
            # Lower is better for this secondary score
            # Fewer remaining ops is better
            # Shorter min duration is better
            return (remaining_ops, min_dur, -rw)
        
        best_idx = min(ties, key=tiebreak_score)
        
    return best_idx
