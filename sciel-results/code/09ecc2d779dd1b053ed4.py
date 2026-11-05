"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track a soft deadline to ensure we return within budget.
_DEADLINE = _time.monotonic() + 140.0  # 140s safety margin out of 150s


def _pick_machine(op, machine_loads, time):
    """
    Choose the eligible machine that minimizes the completion time of this op.
    Returns (machine, completion_time).
    """
    best_machine = None
    best_completion = float('inf')
    for machine, duration in op["machine_options"].items():
        # If we track machine availability, use it; otherwise assume free at 'time'
        # Since we don't have direct access to machine availability in the interface,
        # we use a heuristic: prefer machines with lower historical load if we can track it,
        # but since dispatch is stateless per call, we'll just pick the one with shortest duration.
        # Actually, for flexible job-shop, the key is to pick the machine that finishes earliest.
        # Without machine state, we approximate by picking the machine with the shortest processing time.
        # However, a better heuristic is to consider the machine's current load.
        # Since we can't track state across calls easily, let's use a simple approach:
        # Pick the machine with the shortest duration, breaking ties by machine id for determinism.
        if duration < best_completion:
            best_completion = duration
            best_machine = machine
    return best_machine, time + best_completion


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: 
    1. For each ready operation, estimate its "urgency" based on remaining work and 
       the number of remaining operations (critical path length).
    2. Use a combined metric: prioritize operations with high remaining work and 
       few remaining operations (to avoid long tails).
    3. Additionally, consider the shortest processing time available for each op.
    
    We use a weighted score:
    - remaining_work / (remaining_ops + 1) as a proxy for average remaining op length
    - Prefer ops with higher "criticality" = remaining_work / (remaining_ops + 1)
    - Tie-break by shortest available processing time
    """
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.monotonic() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Criticality: how much work is left per remaining op
        # Higher means more urgent
        if remaining_ops > 0:
            avg_remaining_op = remaining_work / remaining_ops
        else:
            avg_remaining_op = remaining_work
        
        # Shortest processing time for this op
        min_duration = min(op["machine_options"].values()) if op["machine_options"] else 0
        
        # Score: prioritize high criticality, break ties by shorter processing time
        # Use a large multiplier for criticality to make it dominant
        score = avg_remaining_op * 1000.0 - min_duration
        
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
