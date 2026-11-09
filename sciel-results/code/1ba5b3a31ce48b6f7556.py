"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Critical path-based dispatch rule with machine selection.
    
    Strategy:
    1. Compute a "critical path score" for each ready operation:
       - Longest remaining path (sum of minimum durations across remaining ops in job)
       - Weighted by flexibility (number of machine options)
       - Plus a load-balancing term based on machine availability
    2. Select the operation with the highest priority score.
    3. Optionally return (index, machine) where machine minimizes completion time.
    
    The rule aims to:
    - Prioritize jobs on the critical path to reduce makespan
    - Choose machines that minimize completion time (load balancing)
    - Reward flexibility to avoid bottlenecks
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Track elapsed time for safety
    t_start = _time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Compute scores for each ready operation
    scores = np.zeros(n_ops)
    
    # First pass: compute base critical path score
    for i, op in enumerate(ready_operations):
        # Longest remaining path: sum of minimum durations across all remaining ops in this job
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Critical path estimate: use remaining_work as proxy for longest path
        # This is the sum of durations along the critical path
        crit_path = remaining_work
        
        # Flexibility: number of machine options
        machine_options = op.get("machine_options", {})
        n_machines = len(machine_options)
        flexibility = np.log1p(n_machines) if n_machines > 0 else 0.0
        
        # Minimum duration on any machine (for this operation)
        if n_machines > 0:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        
        # Average duration across machines
        if n_machines > 0:
            avg_dur = np.mean(list(machine_options.values()))
        else:
            avg_dur = 0
        
        # Score components:
        # 1. Critical path importance (higher = more urgent)
        w_cp = 0.5
        # 2. Flexibility (more options = less bottleneck risk)
        w_flex = 0.2
        # 3. Shorter average duration preferred (efficiency)
        w_dur = 0.15
        # 4. Remaining ops count (jobs with more remaining work are more critical)
        w_ops = 0.15
        
        score = (
            w_cp * (crit_path / (1.0 + avg_dur)) +  # Normalize by avg duration
            w_flex * flexibility +
            w_dur * (1.0 / (1.0 + avg_dur)) +  # Prefer shorter durations
            w_ops * (remaining_ops / (1.0 + remaining_ops))
        )
        
        scores[i] = score
    
    # Check time
    if _time.time() - t_start > deadline * 0.5:
        # Return best found so far
        return int(np.argmax(scores))
    
    # Machine selection: for the top candidate, pick machine that minimizes completion time
    best_idx = int(np.argmax(scores))
    
    # Try to improve by checking machine selection for top candidates
    top_k = min(3, n_ops)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    best_score = -np.inf
    best_idx_final = best_idx
    best_machine = None
    
    for idx in top_indices:
        op = ready_operations[idx]
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        # For each machine, estimate completion time
        # We don't know machine availability directly, so use a heuristic:
        # Prefer machines with shorter durations (less congestion assumed)
        best_machine_for_op = None
        best_completion = np.inf
        
        for machine, duration in machine_options.items():
            # Heuristic: shorter duration = better completion time
            # Also consider machine load (we don't have this info, so use duration only)
            completion = duration
            if completion < best_completion:
                best_completion = completion
                best_machine_for_op = machine
        
        # Adjust score based on machine selection quality
        adjusted_score = scores[idx]
        if best_machine_for_op is not None:
            # Small bonus for having a good machine option
            adjusted_score += 0.05 * (1.0 / (1.0 + best_completion))
        
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_idx_final = int(idx)
            best_machine = best_machine_for_op
    
    # Check time again
    if _time.time() - t_start > deadline * 0.9:
        return int(best_idx_final)
    
    # Return index (and optionally machine)
    # The interface says "optionally (index, machine)", so return just index for safety
    return int(best_idx_final)
# EVOLVE-BLOCK-END
