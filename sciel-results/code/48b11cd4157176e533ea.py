"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        # Primary: Most remaining work
        rw = op["remaining_work"]
        
        # Secondary: Longest processing time (to clear machine/job)
        # Use the minimum duration across options? Or max?
        # If we assume the simulator picks the best machine, the effective duration is the min duration.
        # But if we want to prioritize "heavy" operations, maybe max duration indicates a big task.
        # Let's use the min duration as the "cost" we are committing to.
        # Actually, LPT suggests picking the longest task.
        # Let's use the max duration in options as a proxy for "big operation".
        durations = list(op["machine_options"].values())
        max_dur = max(durations) if durations else 0
        min_dur = min(durations) if durations else 0
        
        # Tertiary: Rigidity (fewer options is more constrained, prioritize it)
        num_opts = len(op["machine_options"])
        
        # Composite Score
        # We want high rw, high max_dur, low num_opts.
        score = rw * 1000.0 + max_dur - 0.5 * num_opts
        
        scores.append(score)
        
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
