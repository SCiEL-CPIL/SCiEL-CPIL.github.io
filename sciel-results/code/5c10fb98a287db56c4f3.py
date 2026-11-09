"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a refined composite heuristic with smooth surrogate for duration:
      1. Primary: Maximize remaining_work (Longest Remaining Processing Time).
      2. Secondary: Minimize soft-min duration (smooth approximation of min over machines).
      3. Tertiary: Maximize remaining_ops count.
      4. Machine selection: Choose the machine that minimizes duration for the selected op.
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    best_machine_indices = []
    soft_min_durations = []
    hard_min_durations = []
    remaining_works = []
    remaining_ops_counts = []
    job_ids = []
    
    # Temperature parameter for soft-min: lower = closer to hard min
    # Higher = smoother, more balanced consideration of all machines
    soft_min_temp = 2.0
    
    for i, op in enumerate(ready_operations):
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        job = op.get("job", 0)
        
        # Find the machine with shortest duration for this operation
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            machines = list(machine_options.keys())
            
            # Hard min for actual machine selection
            min_idx = durations.index(min(durations))
            best_machine = machines[min_idx]
            best_dur = durations[min_idx]
            
            # Soft-min using log-sum-exp for smooth surrogate
            # soft_min = -1/temp * log(sum(exp(-d/temp)))
            if len(durations) > 1:
                max_neg = max(-d / soft_min_temp for d in durations)
                sum_exp = sum(__import__('math').exp(-d / soft_min_temp - max_neg) for d in durations)
                soft_min = -soft_min_temp * (__import__('math').log(sum_exp) + max_neg)
            else:
                soft_min = best_dur
            
            soft_min_durations.append(soft_min)
            hard_min_durations.append(best_dur)
            best_machine_indices.append(best_machine)
        else:
            best_machine = None
            best_dur = 0
            soft_min_durations.append(0)
            hard_min_durations.append(0)
            best_machine_indices.append(best_machine)
        
        remaining_works.append(mw)
        remaining_ops_counts.append(ro)
        job_ids.append(job)
    
    # Normalize values for composite scoring
    max_rw = max(remaining_works) if remaining_works else 1
    min_soft = min(soft_min_durations) if soft_min_durations else 1
    max_soft = max(soft_min_durations) if soft_min_durations else 1
    max_ro = max(remaining_ops_counts) if remaining_ops_counts else 1
    
    scores = []
    for i in range(n):
        # Normalize remaining work: higher is better
        rw_score = remaining_works[i] / max_rw if max_rw > 0 else 0
        
        # Normalize soft-min duration: shorter is better
        if max_soft > min_soft:
            dur_norm = (soft_min_durations[i] - min_soft) / (max_soft - min_soft)
        else:
            dur_norm = 0.0
        # Smooth penalty: use exponential decay for smooth surrogate
        # This creates a smooth surrogate that favors short operations
        # while being less abrupt than quadratic penalty
        dur_score = __import__('math').exp(-1.5 * dur_norm)
        
        # Normalize remaining ops: higher is better
        ro_score = remaining_ops_counts[i] / max_ro if max_ro > 0 else 0
        
        # Composite score with refined weights
        # Emphasize remaining_work (critical path) most heavily
        # Use exponential duration score for smoothness
        score = 0.55 * rw_score + 0.35 * dur_score + 0.10 * ro_score
        scores.append(score)
    
    # Find the index with maximum score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    # Return (index, machine) to allow explicit machine selection
    if best_machine_indices[best_idx] is not None:
        return (best_idx, best_machine_indices[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
