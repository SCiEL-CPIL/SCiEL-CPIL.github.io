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
    Uses a refined combined heuristic designed to minimize makespan in open-shop scheduling.
    
    Key insights for open-shop makespan minimization:
    1. Prioritize jobs with the most remaining work (they are on the critical path).
    2. Among those, prefer operations that can be done quickly to free up resources sooner.
    3. Jobs with fewer remaining operations are more urgent (closer to completion).
    4. Operations with fewer machine options are more constrained.
    5. Use soft-min approximation for smooth scoring of processing times.
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
    min_rem_work = np.min(all_rem_work) if n > 0 else 0.0
    max_rem_ops = np.max(all_rem_ops) if n > 0 else 1.0
    min_rem_ops = np.min(all_rem_ops) if n > 0 else 0.0
    
    # Range for better normalization
    range_work = max_rem_work - min_rem_work + 1e-9
    range_ops = max_rem_ops - min_rem_ops + 1e-9
    
    # Collect all durations for global soft-min computation
    all_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            all_durations.extend(machine_options.values())
    
    # Use soft-min (log-sum-exp approximation) for smoother scoring
    # soft_min(x) = -log(sum(exp(-x_i / tau)) * tau) for small tau
    tau = 1.0  # temperature parameter for soft-min
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        
        machine_options = op["machine_options"]
        if machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            min_dur = np.min(durations)
            max_dur = np.max(durations)
            avg_dur = np.mean(durations)
            std_dur = np.std(durations) if len(durations) > 1 else 0.0
            num_machines = len(durations)
            
            # Soft-min: smoother approximation of min
            # As tau -> 0, soft_min -> min
            soft_min_dur = -tau * np.log(np.sum(np.exp(-durations / tau)))
        else:
            min_dur = 0.0
            max_dur = 0.0
            avg_dur = 0.0
            std_dur = 0.0
            num_machines = 1
            soft_min_dur = 0.0
        
        # Normalized features (0 to 1 range)
        norm_work = (rem_work - min_rem_work) / range_work
        norm_ops = (rem_ops - min_rem_ops) / range_ops
        
        # Inverse duration score: shorter is better
        # Use 1/(1+x) for bounded, smooth score
        score_short = 1.0 / (soft_min_dur + 1.0)
        
        # Constraint score: fewer machine options = more constrained = prioritize
        score_constraint = 1.0 / (num_machines + 1.0)
        
        # Work-to-time ratio: high means many long operations remaining
        score_ratio = rem_work / (avg_dur + 1.0)
        max_ratio = np.max(all_rem_work) / (np.mean(all_durations) + 1.0) if all_durations else 1.0
        norm_ratio = score_ratio / (max_ratio + 1e-9)
        
        # Urgency: jobs closer to completion (fewer remaining ops) should be prioritized
        # Invert so that fewer remaining ops gets higher score
        urgency = 1.0 - norm_ops
        
        # Critical path indicator: is this job on the critical path?
        # Approximate by comparing remaining work to max
        critical = norm_work ** 2  # Squared to emphasize the most loaded jobs
        
        # Combine heuristics with tuned weights for makespan minimization
        # Primary: remaining work (critical path) - most important
        # Secondary: urgency, speed, constraint
        scores[i] = (
            2.5 * norm_work +       # Most remaining work
            0.8 * critical +         # Emphasize critical path jobs
            0.4 * urgency +          # Fewer remaining ops = more urgent
            0.3 * score_short * 10 + # Shorter processing time
            0.2 * score_constraint * 5 + # More constrained = prioritize
            0.3 * norm_ratio         # Work-to-time ratio
        )
    
    # Tiny random tiebreaker to avoid systematic bias
    scores += np.random.uniform(0, 0.001, size=n)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
