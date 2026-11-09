"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds

def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Composite heuristic with three smooth components:
    1. Soft-min effective duration (LPT surrogate) - smooth approximation of best machine
    2. Most Remaining Work (MRW) - job-level urgency
    3. Throughput/density - remaining_work / effective_duration, rewards quick large ops
    
    We use a soft-min (negative log-sum-exp) to smoothly approximate the minimum
    duration across machine options, which is more robust than hard min and
    provides a smooth surrogate for the nonsmooth min objective.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if too close, fall back to a very fast rule
    if _check_deadline():
        # Fallback: Just pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Temperature for soft-min: lower = closer to hard min
    T = 0.5
    
    soft_min_durations = np.zeros(n)
    remaining_works = np.zeros(n)
    remaining_ops_counts = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            # Soft-min: -log(sum(exp(-d/T)))*T approximates min(d)
            # Use stable form: subtract min before exp
            d_min = np.min(durations)
            # soft_min ≈ d_min - T * log(sum(exp(-(d - d_min)/T)))
            exp_terms = np.exp(-(durations - d_min) / T)
            soft_min_durations[i] = d_min - T * np.log(np.sum(exp_terms))
        else:
            soft_min_durations[i] = 0.0
        remaining_works[i] = op["remaining_work"]
        remaining_ops_counts[i] = op["remaining_ops"]
    
    # Avoid division by zero for density
    eps = 1e-9
    effective_dur = soft_min_durations + eps
    
    # Component 1: LPT surrogate (higher soft-min = longer op = prioritize)
    dur_range = np.ptp(soft_min_durations)
    if dur_range > 0:
        norm_durations = (soft_min_durations - soft_min_durations.min()) / dur_range
    else:
        norm_durations = np.zeros(n)
    
    # Component 2: MRW (higher remaining work = more urgent)
    work_range = np.ptp(remaining_works)
    if work_range > 0:
        norm_works = (remaining_works - remaining_works.min()) / work_range
    else:
        norm_works = np.zeros(n)
    
    # Component 3: Density/throughput = remaining_work / effective_duration
    # Higher density means more work per unit time - prioritize to reduce makespan
    densities = remaining_works / effective_dur
    dens_range = np.ptp(densities)
    if dens_range > 0:
        norm_densities = (densities - densities.min()) / dens_range
    else:
        norm_densities = np.zeros(n)
    
    # Weighted composite score
    # LPT: 0.4, MRW: 0.25, Density: 0.35
    w_lpt = 0.4
    w_mrw = 0.25
    w_density = 0.35
    composite_scores = w_lpt * norm_durations + w_mrw * norm_works + w_density * norm_densities
    
    # Select the operation with the highest composite score
    best_idx = int(np.argmax(composite_scores))
    
    return best_idx
# EVOLVE-BLOCK-END
