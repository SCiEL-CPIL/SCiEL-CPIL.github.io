"""Combined Job-Shop Scheduling Dispatch Rule.

Strategy:
1. Uses a raw weighted sum of features to avoid the instability of batch-dependent normalization.
2. Prioritizes Shortest Processing Time (SPT) using an inverse duration term, which is a strong
   heuristic for clearing machines quickly.
3. Incorporates "Most Remaining Work" (Critical Path) to prioritize jobs that need more time
   to complete, ensuring they start early enough.
4. Incorporates "Most Remaining Operations" to prioritize longer jobs (head start), consistent
   with the higher-scoring current program's direction.
5. Includes a strict deadline guard to prevent timeouts.
"""
# EVOLVE-BLOCK-START
import time as _time

# Global state for tracking evaluation time
_EVAL_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to keep before the 150s limit
_DEADLINE = 150 - _DEADLINE_MARGIN


def _get_deadline():
    """Get or initialize the deadline."""
    global _EVAL_START_TIME
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = _time.time()
    return _EVAL_START_TIME + _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: {machine: duration}
            - remaining_ops: number of remaining operations
            - remaining_work: total remaining work
        
        time: current time
        
    Returns:
        Index of the ready operation to start.
    """
    # Check deadline
    deadline = _get_deadline()
    if _time.time() > deadline:
        # Fallback: return first ready operation
        return 0
    
    if not ready_operations:
        return 0
    
    # If only one option, return it
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    best_idx = 0
    best_score = float('-inf')
    
    # Weights tuned to balance SPT, Critical Path (Work), and Job Length (Ops)
    # Based on analysis:
    # - SPT (1/dur) needs to be scaled to be comparable to Work.
    # - Work (remaining_work) is a strong signal for Critical Path.
    # - Ops (remaining_ops) favors longer jobs (head start).
    
    W_SPT = 10.0       # Weight for Shortest Processing Time (inverse duration)
    W_WORK = 0.2       # Weight for Remaining Work
    W_OPS = 1.0        # Weight for Remaining Ops
    
    for i in range(n):
        op = ready_operations[i]
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        # Calculate min duration
        min_dur = min(machine_options.values())
        
        # Get remaining work and ops
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Calculate score
        # 1. SPT component: Higher is better (shorter duration)
        #    Using +1.0 to avoid division by zero and smooth the curve
        spt_score = W_SPT / (min_dur + 1.0)
        
        # 2. Work component: Higher is better (more work left = critical path)
        work_score = W_WORK * remaining_work
        
        # 3. Ops component: Higher is better (more ops left = longer job, start early)
        ops_score = W_OPS * remaining_ops
        
        score = spt_score + work_score + ops_score
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
