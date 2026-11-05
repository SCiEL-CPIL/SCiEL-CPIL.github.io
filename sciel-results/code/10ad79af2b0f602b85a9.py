"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = _time.perf_counter()
    return _time.perf_counter() - _START


def dispatch(ready_operations, time):
    """
    Machine-load balancing dispatch rule for open-shop scheduling.
    
    Strategy: For each ready operation, compute a score that balances:
      1. Remaining work on the operation (urgency)
      2. Machine constraint: prefer operations whose best machine option has
         lower implied load (fewer other ops competing / shorter duration)
      3. Remaining ops count (jobs with more ops left are less urgent per-op)
    
    We select the operation with the highest composite score.
    """
    global _START
    if _START is None:
        _START = _time.perf_counter()
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Build scoring vectors
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, find the minimum duration among its machine options
    # and the number of machine options (flexibility)
    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    num_options = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        durs = np.array(list(opts.values()), dtype=np.float64)
        min_durations[i] = np.min(durs) if len(durs) > 0 else 0.0
        max_durations[i] = np.max(durs) if len(durs) > 0 else 0.0
        num_options[i] = float(len(durs))
    
    # Normalize features to [0, 1] range for weighting
    rw_max = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    ro_max = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    xd_max = np.max(max_durations) if np.max(max_durations) > 0 else 1.0
    
    # Score components:
    # 1. Remaining work (higher = more urgent) - weight 0.4
    score_rw = remaining_works / rw_max * 0.4
    
    # 2. Machine constraint: lower min duration on best machine is better
    #    (less time blocking that machine) - weight 0.25
    score_machine = (1.0 - min_durations / md_max) * 0.25
    
    # 3. Fewer remaining ops means this job is closer to completion,
    #    finishing it frees up resources sooner - weight 0.15
    score_ops = (1.0 - remaining_ops / ro_max) * 0.15
    
    # 4. Flexibility: more machine options means less bottleneck risk - weight 0.1
    no_max = np.max(num_options) if np.max(num_options) > 0 else 1.0
    score_flex = (num_options / no_max) * 0.1
    
    # 5. Penalty for high max duration (longest possible block) - weight 0.1
    score_dur = (1.0 - max_durations / xd_max) * 0.1
    
    total_score = score_rw + score_machine + score_ops + score_flex + score_dur
    
    # Small random tiebreaker to avoid systematic bias
    rng = np.random.default_rng(seed=int(time * 1000) % (2**31))
    tiebreaker = rng.random(n) * 0.001
    
    total_score += tiebreaker
    
    return int(np.argmax(total_score))
# EVOLVE-BLOCK-END
