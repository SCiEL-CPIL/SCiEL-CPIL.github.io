"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard: 150s wall-clock kill; keep a safety margin.
_DEADLINE = 150.0
_START_TIME = _time.time()
_MARGIN = 10.0  # seconds of safety margin


def _elapsed_ok():
    return (_time.time() - _START_TIME) < (_DEADLINE - _MARGIN)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a multi-factor heuristic.
    
    Factors considered:
    1. remaining_work: total work left for the job (higher = more urgent)
    2. remaining_ops: number of ops left (higher = more complex, start early)
    3. Machine flexibility: how many machines the op can run on (fewer = more constrained, prioritize)
    4. Min duration: shorter ops can be squeezed in (SPT-like benefit)
    5. Machine load balance: prefer machines that are less loaded
    
    We use a weighted score combining these factors. The weight for
    remaining_work is dominant since it directly correlates with makespan.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Quick path for small n
    if n == 1:
        return 0
    
    if not _elapsed_ok():
        # Fallback to simple rule if we're running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Precompute factors
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        # Machine flexibility: fewer options = more constrained
        num_machines = len(mo)
        flexibility = 1.0 / max(num_machines, 1)
        
        # Min duration across available machines
        min_dur = min(mo.values()) if mo else 0
        
        # Max duration (for load estimation)
        max_dur = max(mo.values()) if mo else 0
        
        # Score components:
        # 1. Remaining work (dominant) - normalized by max possible
        # 2. Remaining ops (secondary urgency)
        # 3. Flexibility (constrained ops first)
        # 4. Shortest processing time (SPT effect)
        
        # We want to maximize urgency, so:
        # - Higher remaining_work -> higher score
        # - Higher remaining_ops -> higher score  
        # - Lower flexibility (fewer machines) -> higher score
        # - Lower min_dur -> higher score (SPT)
        
        scores[i] = (
            1.0 * rw +           # dominant: most work remaining
            0.3 * ro +           # secondary: more ops to complete
            2.0 * flexibility +  # constrained ops get priority
            0.1 / (min_dur + 1.0)  # SPT: shorter ops slightly preferred
        )
    
    # Select the operation with highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
