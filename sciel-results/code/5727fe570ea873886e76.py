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


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a combined heuristic designed to minimize makespan in open-shop scheduling.
    
    Key insights for open-shop makespan minimization:
    1. Prioritize jobs with the most remaining work (they are the critical path).
    2. Among those, prefer operations that can be done quickly (short min duration)
       to free up resources sooner.
    3. Jobs with fewer remaining operations are more urgent.
    4. Operations with fewer machine options are more constrained.
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
    all_rem_work = np.array([op["remaining_work"] for op in ready_operations])
    all_rem_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    max_rem_work = np.max(all_rem_work) if n > 0 else 1.0
    max_rem_ops = np.max(all_rem_ops) if n > 0 else 1.0
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_dur = min(durations)
            max_dur = max(durations)
            avg_dur = sum(durations) / len(durations)
            num_machines = len(durations)
        else:
            min_dur = 0.0
            max_dur = 0.0
            avg_dur = 0.0
            num_machines = 1
        
        # Normalized features
        norm_work = rem_work / (max_rem_work + 1e-9)
        norm_ops = rem_ops / (max_rem_ops + 1e-9)
        
        # Shorter minimum processing time is better (frees machine sooner)
        # Use inverse with scaling
        score_short = 1.0 / (min_dur + 1.0)
        
        # Fewer machine options = more constrained = prioritize
        score_constraint = 1.0 / (num_machines + 1.0)
        
        # Ratio: remaining work relative to average processing time
        # High ratio means this job has a lot of work relative to how long each op takes
        score_ratio = rem_work / (avg_dur + 1.0)
        
        # Combine heuristics with weights tuned for makespan minimization
        # Primary: most remaining work (critical path)
        # Secondary: urgency (fewer ops left), speed (short min dur), constraint
        scores[i] = (
            2.0 * norm_work +
            0.5 * norm_ops +
            0.3 * score_short * 10 +
            0.2 * score_constraint * 5 +
            0.4 * score_ratio / (max(all_rem_work) + 1.0)
        )
    
    # Tiny random tiebreaker
    scores += np.random.uniform(0, 0.001, size=n)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
