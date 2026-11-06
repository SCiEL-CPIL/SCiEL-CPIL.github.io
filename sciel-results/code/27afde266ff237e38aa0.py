"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.perf_counter()
_DEADLINE_LIMIT = 110.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    if time_module.perf_counter() - _START_TIME > _DEADLINE_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # For each operation, get the minimum duration among machine options
    # This is the key structural insight: in open-shop, the machine choice
    # is flexible, so the effective processing time is the minimum available
    min_durs = np.empty(n, dtype=float)
    max_durs = np.empty(n, dtype=float)
    mean_durs = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if durations:
            min_durs[i] = min(durations)
            max_durs[i] = max(durations)
            mean_durs[i] = np.mean(durations)
        else:
            min_durs[i] = 0.0
            max_durs[i] = 0.0
            mean_durs[i] = 0.0

    # Min-max normalize each feature to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    rw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durs)
    xd_norm = normalize(max_durs)
    mn_norm = normalize(mean_durs)

    # Structural change: use a critical-path-inspired priority that
    # emphasizes the job with the most remaining work AND the operation
    # that can be completed fastest (shortest min duration).
    #
    # The key insight from open-shop scheduling: the makespan is dominated
    # by the job with the most total remaining work. However, starting an
    # operation with a short processing time on an available machine can
    # reduce idle time and keep the schedule tight.
    #
    # We use a multiplicative interaction: jobs with high remaining work
    # get a boost if their current operation can finish quickly, creating
    # a "critical path with flexibility" heuristic.
    
    # Base priority: remaining work dominates
    base = 0.55 * rw_norm + 0.20 * ro_norm
    
    # Flexibility bonus: shorter operations on flexible machines
    # Use (1 - normalized) so shorter durations get higher bonus
    flex_bonus = 0.15 * (1.0 - md_norm) + 0.10 * (1.0 - mn_norm)
    
    # Interaction term: reward high-work jobs that also have short ops
    # This creates a nonlinear preference that pure linear weighting misses
    interaction = 0.10 * rw_norm * (1.0 - md_norm)
    
    score = base + flex_bonus + interaction

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
