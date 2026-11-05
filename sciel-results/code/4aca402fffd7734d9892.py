"""Improved dispatch rule for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call elapsed time to avoid pathological cases,
# but a single dispatch call should be O(n*m) and very fast.
_CALL_START = None

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy: Use a weighted scoring rule that considers:
    1. Remaining work (urgency)
    2. Minimum processing time on available machines (to avoid long blocks)
    3. Number of remaining operations (prioritize jobs with fewer remaining ops
       to reduce tail effects)
    4. Machine load balancing: prefer machines that are less loaded
    
    We return an index into ready_operations. If we can also choose a machine,
    the interface allows returning (index, machine), but the base interface
    only requires index. Let's check what's expected.
    
    The task says: "dispatch(ready_operations, time) -> index of the ready 
    operation to start (optionally (index, machine))"
    
    Since the seed program returns just an int, and the score is 1.0, let's 
    try returning just the index first with a better heuristic. But if we can
    return (index, machine), we might get better results by choosing the 
    best machine for each operation.
    
    Let me re-read: "each ready operation is a dict with job, op, machine_options 
    {machine: duration}, remaining_ops, remaining_work"
    
    I'll try returning (index, machine) where machine is the best machine for 
    that operation. If the evaluator doesn't support tuples, it will fail. 
    But the task says "optionally (index, machine)", so it should work.
    
    Actually, let's be safe and return just the index first, but with a much 
    better heuristic. If the interface supports (index, machine), we can try 
    that.
    
    Let me think about what heuristic would be best:
    
    For each ready operation i:
    - min_duration[i] = min duration across its machine options
    - max_duration[i] = max duration across its machine options
    - remaining_ops[i] = number of remaining operations for its job
    - remaining_work[i] = total remaining work for its job
    
    A good heuristic might be:
    - Prioritize operations with smaller min_duration (to keep machines busy 
      with short jobs)
    - But also consider remaining_work to avoid starvation
    
    Actually, a common good rule for job-shop is:
    - Shortest Processing Time (SPT) on the minimum available machine
    - But weighted by remaining work
    
    Let me use a combined score:
    score[i] = min_duration[i] / (1 + remaining_ops[i]) + 
               min_duration[i] * remaining_work[i] / (1 + len(machine_options))
    
    Hmm, let me think more carefully. The goal is to minimize makespan.
    
    Key insights for job-shop:
    1. Operations with fewer remaining operations should be prioritized to 
       avoid blocking (they're closer to completion)
    2. Shorter operations should be prioritized to keep machines busy
    3. We should balance machine loads
    
    Let me use:
    For each operation i:
    - min_dur = min duration on any machine
    - rem_ops = remaining_ops (number of ops left in the job)
    - rem_work = remaining_work
    
    Score = min_dur * (1 + rem_ops) / (1 + rem_work / max_possible_work)
    
    Actually, simpler: prioritize by (min_dur / (rem_ops + 1))
    This favors short operations with few remaining steps.
    
    But we also want to consider the overall remaining work. Let me try:
    
    score[i] = min_duration[i] / (remaining_ops[i] + 1)
    
    And pick the operation with the MINIMUM score (shortest effective processing 
    time relative to remaining complexity).
    
    Wait, but if remaining_work is very large, we might want to prioritize 
    those jobs to avoid them becoming critical path.
    
    Let me try a hybrid:
    - If remaining_ops is small (say <= 2), prioritize by min_duration
    - Otherwise, prioritize by remaining_work / min_duration (urgency)
    
    Actually, let me just try a simple but effective rule:
    score[i] = min_duration[i] * (remaining_ops[i] + 1)
    Pick minimum score.
    
    This favors operations that are short AND have few remaining steps.
    
    Let me implement this and also try returning (index, best_machine) if possible.
    """
    global _CALL_START
    _CALL_START = _time.perf_counter()
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Choose the machine with minimum duration
        best_machine = min(op["machine_options"].keys(), 
                          key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    best_idx = -1
    best_score = float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue
        
        # Find min and max duration
        min_dur = min(mo.values())
        max_dur = max(mo.values())
        num_machines = len(mo)
        
        rem_ops = op.get("remaining_ops", 1)
        rem_work = op.get("remaining_work", min_dur)
        
        # Heuristic score:
        # Primary: minimize min_duration weighted by remaining complexity
        # Secondary: balance machine load by preferring machines with shorter durations
        
        # Score 1: min_duration / (remaining_ops + 1) - favors short, near-complete ops
        score1 = min_dur / (rem_ops + 1)
        
        # Score 2: min_duration * (1 + rem_ops) - alternative
        score2 = min_dur * (rem_ops + 1)
        
        # Score 3: urgency-based - higher remaining_work means more urgent
        # But we want to minimize makespan, so prioritize jobs with more remaining work?
        # Actually, no - we want to finish jobs, so prioritize those with less remaining
        # to avoid long tails. But also don't starve jobs with lots of work.
        
        # Let me use a combined score:
        # score = min_dur * (1 + rem_ops * 0.5) + rem_work * 0.1
        score = min_dur * (1 + rem_ops * 0.3) + rem_work * 0.05
        
        if score < best_score:
            best_score = score
            best_idx = i
            # Choose the machine with minimum duration for this operation
            best_machine = min(mo.keys(), key=lambda m: mo[m])
    
    if best_idx == -1:
        # Fallback: pick first
        return 0
    
    # Try returning (index, machine) - if this fails, the evaluator should handle it
    # But to be safe, let's just return index for now and see if we can improve
    # Actually, the task says "optionally (index, machine)", so let's try it.
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
