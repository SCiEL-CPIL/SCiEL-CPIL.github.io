"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: we must return well before the evaluator kills us (150s wall-clock).
# We track a per-call budget so that if we ever do heavy work we can bail out early.
# For this dispatch rule, the work is trivial, but we keep the guard for safety.
_DEADLINE_SECONDS = 120.0  # safety margin below the 150s kill
_CALL_START = None
_CALL_BUDGET = 0.1  # seconds per dispatch call


def _elapsed():
    global _CALL_START
    if _CALL_START is None:
        _CALL_START = _time.monotonic()
    return _time.monotonic() - _CALL_START


def _check_deadline(fallback_index):
    """If we've exceeded our per-call budget, return the fallback index."""
    if _elapsed() > _CALL_BUDGET:
        return fallback_index
    return None


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Combined rule that balances:
    1. Most Work Remaining (MWR) - prioritize jobs with more total work left
    2. Shortest Processing Time (SPT) - on the chosen machine, pick shortest duration
    3. Machine load balancing - avoid overloading the most congested machines
    4. Remaining operations count - jobs with fewer remaining ops may be closer to completion
    
    We use a weighted scoring approach:
    - Primary: maximize remaining_work (longest job remaining)
    - Secondary: minimize processing time on the selected machine (SPT)
    - Tertiary: prefer machines with lower current load (if we can infer it)
    
    Since we don't have direct access to machine loads, we use a heuristic:
    For each ready operation, compute a composite score.
    """
    global _CALL_START
    _CALL_START = _time.monotonic()
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Quick deadline check
    fb = _check_deadline(0)
    if fb is not None:
        return fb
    
    # For each ready operation, determine the best machine and compute a score
    # machine_options: {machine: duration}
    # We want to:
    #   - Pick the operation that, when scheduled, leads to the best makespan
    #   - Heuristic: combine remaining_work, min processing time, and remaining_ops
    
    scores = np.zeros(n)
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration across available machines
    # and also consider the number of machine options (more options = more flexibility)
    min_durations = np.zeros(n)
    mean_durations = np.zeros(n)
    num_options = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            min_durations[i] = min(durations)
            mean_durations[i] = np.mean(durations)
            num_options[i] = len(durations)
        else:
            min_durations[i] = 0
            mean_durations[i] = 0
            num_options[i] = 1
    
    # Normalize features to [0, 1] for weighting
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_min_dur = normalize(min_durations)  # lower is better, so we'll invert
    norm_mean_dur = normalize(mean_durations)
    
    # Composite score:
    # We want to MAXIMIZE:
    #   - remaining_work (jobs with more work left should go first to reduce tail)
    #   - (1 - min_duration) (shorter processing time is better)
    #   - (1 - mean_duration) (shorter average processing time)
    #   - (1 - remaining_ops) (fewer remaining ops = closer to completion, less blocking)
    #   - num_options (more machine choices = more flexibility, slight bonus)
    
    # Weights based on known JSS heuristics:
    # MWR is strong; SPT is strong; machine load balancing via flexibility
    w_work = 0.35
    w_spt = 0.30
    w_ops = 0.15
    w_flex = 0.10
    w_mean_dur = 0.10
    
    scores = (
        w_work * norm_work +
        w_spt * (1.0 - norm_min_dur) +
        w_ops * (1.0 - norm_ops) +
        w_flex * normalize(num_options) +
        w_mean_dur * (1.0 - norm_mean_dur)
    )
    
    # Check deadline again
    fb = _check_deadline(0)
    if fb is not None:
        return fb
    
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
