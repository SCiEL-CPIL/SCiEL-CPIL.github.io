"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard to ensure we finish within the 120s limit
# with a safety margin. We track the start time of the entire evaluation
# (or the first call, if the evaluator calls dispatch multiple times per instance).
# Since the evaluator runs 5 instances, we need to be careful.
# However, the dispatch function is called many times per instance.
# A better approach is to assume the function is fast enough that it doesn't
# hit the 120s limit on its own, but we add a guard just in case.
# Actually, the prompt says "The evaluator KILLS the program after 150s wall-clock."
# and "Each evaluation must finish within 120 s."
# The dispatch function is called repeatedly. If we do heavy computation per call,
# it might be slow. But the problem size is small (10 jobs, 5 machines).
# So a simple heuristic should be very fast.
# Let's implement a more sophisticated heuristic.

# We will use a weighted sum of:
# 1. Remaining work (urgency)
# 2. Critical path tail (estimated completion time if started now)
# 3. Machine load (prefer less loaded machines)
# 4. Processing time on the chosen machine (prefer shorter)

def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next and optionally the machine.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: dict {machine: duration}
            - remaining_ops: list of remaining operation ids (or count?)
            - remaining_work: total remaining processing time for this job
        time: current simulation time
        
    Returns:
        index of the ready operation to start, or (index, machine)
    """
    if not ready_operations:
        return 0  # Should not happen, but safe fallback
    
    n = len(ready_operations)
    if n == 1:
        # Only one choice, pick the best machine for it
        op = ready_operations[0]
        if op["machine_options"]:
            best_mach = min(op["machine_options"], key=lambda m: op["machine_options"][m])
            return (0, best_mach)
        return 0
    
    # We'll score each ready operation
    # For each operation, we need to estimate the "cost" of scheduling it now.
    # A good heuristic is to minimize the estimated completion time of the job.
    
    # First, let's compute some global stats if possible.
    # We don't have global machine load info directly, but we can infer from the problem.
    # Actually, we only have the ready operations. We don't know the load of machines.
    # But we can use the machine_options to estimate.
    
    # Let's try a different approach: 
    # For each ready operation i, and each machine m in its options:
    #   score(i, m) = remaining_work[i] + duration[i][m] + (some penalty for machine m)
    # We want to minimize the score.
    
    # But we don't have machine load. Let's assume that choosing a machine with shorter
    # duration is better, and also that we should prioritize jobs with more remaining work.
    
    # Let's use a weighted combination:
    # score = w1 * remaining_work + w2 * best_duration + w3 * (1 / best_duration)
    # Actually, let's think about critical path.
    # The critical path of a job is the sum of the minimum processing times of its remaining ops.
    # We have remaining_work, which is the sum of remaining processing times.
    # But we don't have the min times per op.
    
    # Let's use a simpler but effective heuristic:
    # 1. For each ready operation, find the best machine (shortest duration).
    # 2. Compute a priority score for the operation.
    # 3. Pick the operation with the highest priority.
    
    # Priority score components:
    # - remaining_work: higher is more urgent
    # - best_duration: lower is better (can finish sooner)
    # - We can also consider the ratio: remaining_work / best_duration (urgency per unit time)
    
    # Let's try: score = remaining_work * (1 + 1/best_duration)
    # This gives higher score to jobs with more work and shorter processing time.
    
    # But we also need to pick a machine. Let's pick the machine with the shortest duration.
    # Or, if we can return (index, machine), we should.
    
    # Let's compute for each operation:
    scores = []
    best_machines = []
    
    for op in ready_operations:
        mach_opts = op["machine_options"]
        if not mach_opts:
            # No machine options? Should not happen.
            scores.append(0)
            best_machines.append(None)
            continue
        
        # Find the machine with the shortest duration
        best_mach = min(mach_opts, key=lambda m: mach_opts[m])
        best_dur = mach_opts[best_mach]
        
        # Compute a score. We want to maximize this.
        # Idea: prioritize jobs that are "close to done" in terms of critical path,
        # but also have significant remaining work.
        
        # Let's use: score = remaining_work / (1 + best_dur)
        # This favors jobs with more work but penalizes long processing times.
        
        # Or: score = remaining_work * (1 / best_dur)
        # This favors jobs with high work-to-time ratio.
        
        # Let's try a blend:
        # score = 0.7 * (remaining_work / max(1, best_dur)) + 0.3 * remaining_work
        
        rem_work = op["remaining_work"]
        
        # Normalize? We don't know the scale. Let's use raw values.
        # If remaining_work is 0, score is 0.
        
        score = rem_work / (1.0 + best_dur) + 0.5 * rem_work
        scores.append(score)
        best_machines.append(best_mach)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Return (index, machine) if we have a best machine
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
