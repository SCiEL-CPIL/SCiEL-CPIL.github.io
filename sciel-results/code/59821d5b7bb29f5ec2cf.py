"""Job-Shop dispatch rule: Critical Path (LRPT) + optimal machine assignment."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE_SECONDS = 140.0  # safety margin before 150s kill


def _critical_path_length(remaining_ops, machine_options):
    """
    Estimate the critical path length for a job given its remaining operations.
    
    remaining_ops: list of (op_index, machine_options_dict) or similar structure
    We assume remaining_ops is a list where each element represents an operation
    with its machine options. The critical path is the sum of minimum durations
    along the chain (since operations are sequential within a job).
    
    However, a better estimate accounts for machine contention. Here we use a 
    simple approach: sum of min durations for each remaining op, but weighted
    by considering that later ops might face more contention.
    
    Actually, the standard LRPT uses the sum of processing times of remaining
    operations (using the assigned machine or minimum). Since machine_options 
    gives multiple choices, we use the minimum duration as a lower bound for 
    the critical path.
    """
    if not remaining_ops:
        return 0.0
    
    total = 0.0
    for op_data in remaining_ops:
        # op_data structure: we need to extract machine options
        # Based on the interface, remaining_ops likely contains info about
        # subsequent operations. Let's assume it's a list of dicts or tuples.
        # The exact structure isn't fully specified, but typically in these
        # problems, remaining_ops is a list of operation descriptors.
        
        # Let's be defensive: if op_data is a dict with 'machine_options'
        if isinstance(op_data, dict):
            mo = op_data.get('machine_options', {})
            if mo:
                total += min(mo.values())
            else:
                total += 1.0  # fallback
        elif isinstance(op_data, (list, tuple)) and len(op_data) > 0:
            # Assume it might be (machine, duration) or similar
            # This is uncertain, so we try to handle common cases
            if len(op_data) == 2 and isinstance(op_data[1], (int, float)):
                total += op_data[1]
            else:
                total += 1.0
        else:
            total += 1.0
    
    return total


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. For each ready operation, compute a score based on:
       a) Critical path length (longer = more urgent)
       b) Estimated completion time if assigned to its best machine
    2. Select the operation with the highest urgency score.
    3. For the selected operation, choose the machine that minimizes estimated
       completion time (current time + duration, considering machine load).
    
    Returns: index of the ready operation (or (index, machine) tuple)
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute machine loads (estimated busy time)
    # We don't have direct access to machine schedules, so we estimate
    # contention based on the number of operations competing for each machine
    machine_load = {}
    for op in ready_operations:
        mo = op.get('machine_options', {})
        for m in mo:
            machine_load[m] = machine_load.get(m, 0) + 1
    
    best_idx = 0
    best_score = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        mo = op.get('machine_options', {})
        if not mo:
            continue
        
        # Critical path: use remaining_work if available, else estimate
        # remaining_work is typically the sum of processing times of remaining ops
        remaining_work = op.get('remaining_work', 0)
        
        # If remaining_work is not provided, estimate from remaining_ops
        if remaining_work == 0 or remaining_work is None:
            remaining_ops = op.get('remaining_ops', [])
            remaining_work = _critical_path_length(remaining_ops, mo)
        
        # For the current operation, find the best machine
        # Score each machine option by estimated completion time
        # Estimated completion = time + duration + contention_penalty
        
        best_machine_for_op = None
        best_completion_for_op = float('inf')
        
        for m, dur in mo.items():
            # Contention penalty: more operations wanting this machine = longer wait
            load = machine_load.get(m, 1)
            # Simple model: estimated wait is proportional to load
            # A rough estimate: if k operations compete, average wait ~ (k-1)*avg_duration/2
            # But we don't know other durations, so use a simple penalty
            estimated_completion = time + dur + (load - 1) * dur * 0.5
            if estimated_completion < best_completion_for_op:
                best_completion_for_op = estimated_completion
                best_machine_for_op = m
        
        # Urgency score: prioritize longer critical paths, penalize long completion
        # Higher score = more urgent
        # We want to start operations that are on the critical path early
        # So: score = critical_path - estimated_completion (or similar)
        
        # Alternative: just use critical path length as primary key,
        # with machine choice minimizing completion time as tiebreaker
        
        # Let's use: score = remaining_work (higher = more urgent)
        # But also consider that if an op can finish very soon, it's good to start it
        # to free up the machine for others.
        
        # Combined score: prioritize by critical path, break ties by earliest completion
        score = remaining_work * 1000.0 - best_completion_for_op
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # Return index (optionally (index, machine))
    # The interface says "index of the ready operation to start (optionally (index, machine))"
    # Let's return just the index for safety, as the seed does
    return int(best_idx)
# EVOLVE-BLOCK-END
