"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0
_SOFT_MARGIN = 5.0

def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _soft_min(durations, temperature=1.0):
    """
    Soft-min using log-sum-exp trick for smooth surrogate of min.
    As temperature -> 0, approaches true min.
    """
    if len(durations) == 0:
        return 0.0
    d = np.array(durations, dtype=np.float64)
    if len(d) == 1:
        return float(d[0])
    # Use negative log-sum-exp for soft min
    d_min = np.min(d)
    shifted = d - d_min
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return float(d_min - temperature * log_sum)


def _soft_max(durations, temperature=1.0):
    """
    Soft-max using log-sum-exp trick for smooth surrogate of max.
    As temperature -> 0, approaches true max.
    """
    if len(durations) == 0:
        return 0.0
    d = np.array(durations, dtype=np.float64)
    if len(d) == 1:
        return float(d[0])
    d_max = np.max(d)
    shifted = d - d_max
    log_sum = np.log(np.sum(np.exp(shifted / temperature)))
    return float(d_max + temperature * log_sum)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a refined combined heuristic with soft-min/soft-max surrogates
    designed to minimize makespan in open-shop scheduling.
    
    Key insights for open-shop makespan minimization:
    1. Prioritize jobs with the most remaining work (critical path).
    2. Among those, prefer operations with short expected completion time.
    3. Use soft-min for smooth machine duration estimation.
    4. Jobs with fewer remaining operations are more urgent.
    5. Balance machine utilization across the shop.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # If close to deadline, use fastest simple rule
    if _check_deadline():
        # Most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    # Pre-compute global stats for normalization
    all_rem_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    all_rem_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    max_rem_work = np.max(all_rem_work) if n > 0 else 1.0
    min_rem_work = np.min(all_rem_work) if n > 0 else 0.0
    range_work = max_rem_work - min_rem_work + 1e-9
    
    max_rem_ops = np.max(all_rem_ops) if n > 0 else 1.0
    min_rem_ops = np.min(all_rem_ops) if n > 0 else 0.0
    range_ops = max_rem_ops - min_rem_ops + 1e-9
    
    # Temperature for soft-min/soft-max - lower means sharper
    soft_temp = 0.5
    
    for i, op in enumerate(ready_operations):
        rem_work = float(op["remaining_work"])
        rem_ops = float(op["remaining_ops"])
        
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_dur = _soft_min(durations, soft_temp)
            max_dur = _soft_max(durations, soft_temp)
            avg_dur = float(np.mean(durations))
            num_machines = len(durations)
            # Expected completion time: use soft-min as optimistic estimate
            expected_time = min_dur
        else:
            min_dur = 0.0
            max_dur = 0.0
            avg_dur = 0.0
            num_machines = 1
            expected_time = 0.0
        
        # Normalized features (0 to 1 range)
        norm_work = (rem_work - min_rem_work) / range_work
        norm_ops = (rem_ops - min_rem_ops) / range_ops
        
        # Criticality: remaining work per remaining op (average work per op)
        # High values mean this job has heavy operations remaining
        work_per_op = rem_work / (rem_ops + 1e-9)
        
        # Normalize work_per_op
        all_wpo = np.array([float(op2["remaining_work"]) / (float(op2["remaining_ops"]) + 1e-9) 
                            for op2 in ready_operations])
        max_wpo = np.max(all_wpo) + 1e-9
        norm_wpo = work_per_op / max_wpo
        
        # Speed score: shorter expected time is better
        # Use inverse with scaling for smoothness
        max_expected = max_dur + 1.0 if max_dur > 0 else avg_dur + 1.0
        score_speed = 1.0 / (expected_time + 1.0)
        norm_speed = score_speed / (1.0 / (1.0))  # already in (0,1] range
        
        # Constraint score: fewer machine options = more constrained
        score_constraint = 1.0 / (num_machines + 1.0)
        
        # Urgency: jobs with fewer remaining ops should be prioritized
        # to avoid blocking
        urgency = 1.0 - norm_ops
        
        # Balance: consider both work remaining and time to complete
        # A job with high work but short ops is more urgent than one with
        # low work but long ops
        completion_estimate = rem_work / (avg_dur + 1e-9)
        
        # Combine heuristics with tuned weights
        # Primary: remaining work (critical path contribution)
        # Secondary: speed of current op (frees resources)
        # Tertiary: urgency (fewer ops left)
        # Quaternary: constraint (limited machine options)
        # Quinary: work-per-op intensity
        scores[i] = (
            1.5 * norm_work +
            0.8 * norm_speed +
            0.4 * urgency +
            0.3 * score_constraint +
            0.5 * norm_wpo
        )
    
    # Deterministic selection - no random tiebreaker for consistent scoring
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
