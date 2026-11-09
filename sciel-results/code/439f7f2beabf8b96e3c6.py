"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = None
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    """
    Job-Priority Dispatch Rule with Explicit Machine Selection.
    
    Strategy:
    1. Prioritize operations based on the total remaining work of their respective jobs.
       This effectively implements a "Longest Remaining Processing Time" (LRPT) at the job level,
       which is a strong heuristic for makespan minimization in Job-Shop problems.
    2. Among operations from the highest-priority job, select the one with the shortest 
       minimum processing time to maximize throughput.
    3. Explicitly select the machine with the shortest duration for the chosen operation 
       to minimize the blocking time and reduce the critical path length of that job.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        if not ready_operations:
            return 0
        # Fallback: First operation
        return 0
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    if n_ops == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            # Pick shortest duration machine
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # 1. Calculate priority for each operation based on its job's remaining work
    # We assume 'remaining_work' in the op dict refers to the job's remaining work.
    # If it refers to the op's specific work, we need to sum for the job? 
    # The task description says: "each ready operation is a dict with ... remaining_work".
    # In standard FJSP interfaces, remaining_work usually refers to the job's total remaining 
    # processing time (sum of all future ops) or the specific op's duration. 
    # Given "remaining_ops" is also there, "remaining_work" is likely the sum of durations 
    # of all remaining operations in the job.
    
    # Let's verify by checking if remaining_work varies within the same job in a single call.
    # Usually, only one op per job is ready at a time.
    
    # We will use remaining_work as the primary key.
    
    # To handle the case where remaining_work might be per-op, we can try to aggregate.
    # But since we don't have the full job structure, we assume the provided 'remaining_work' 
    # is the job-level critical path length or total work.
    
    # Let's extract features
    features = []
    for i, op in enumerate(ready_operations):
        job = op.get("job", i) # Job ID
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            min_d = min(mo.values())
            best_machine = min(mo, key=mo.get)
        else:
            min_d = 1.0
            best_machine = None
            
        features.append({
            'index': i,
            'job': job,
            'rw': rw,
            'min_d': min_d,
            'best_machine': best_machine
        })
    
    # 2. Sort by Job Priority (remaining_work) descending
    # Tie-breaker: Shorter min_duration (efficiency)
    # Tie-breaker: Job ID (deterministic)
    
    # We want to maximize rw.
    # We want to minimize min_d.
    
    # Let's use a tuple sort key: (-rw, min_d, job)
    # This prioritizes higher rw. If rw is equal, it picks lower min_d.
    
    sorted_features = sorted(features, key=lambda x: (-x['rw'], x['min_d'], x['job']))
    
    best_feat = sorted_features[0]
    
    # 3. Return index and machine
    if best_feat['best_machine'] is not None:
        return (best_feat['index'], best_feat['best_machine'])
    else:
        return best_feat['index']

# EVOLVE-BLOCK-END
