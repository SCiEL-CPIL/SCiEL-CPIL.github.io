"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation using a Longest Processing Time (LPT)
    heuristic, with tie-breaking by Most Work Remaining (MWR).
    
    Rationale:
    1. LPT prioritizes operations that require the longest processing time on any
       available machine. This helps prevent long jobs from being delayed by
       shorter jobs, which can lead to higher makespans due to idle time or
       congestion on critical machines.
    2. Ties are broken by selecting the operation with the most remaining work
       in its job, ensuring that critical path jobs are still progressed.
    
    This approach contrasts with the previous MWR-dominant strategy by focusing
    on immediate resource commitment duration rather than total future load.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # LPT: Maximize the maximum duration available
        # If no machine options (should not happen in valid JSS), default to 0
        max_duration = max(machine_options.values()) if machine_options else 0
        
        # Primary: LPT (higher is better)
        # Secondary: MWR (higher is better)
        # We scale max_duration heavily to ensure it is the primary decision factor
        score = max_duration * 10000.0 + remaining_work
        
        scores.append(score)
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
