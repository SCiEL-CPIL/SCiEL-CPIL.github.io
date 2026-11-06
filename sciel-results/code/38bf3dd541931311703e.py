import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    # Deadline guard
    start_t = time_module.perf_counter()
    DEADLINE = 145.0  # 145s to leave 5s margin before 150s kill

    def elapsed():
        return time_module.perf_counter() - start_t

    # Heuristic scoring function
    # We use a composite score that balances:
    # 1. Remaining Work (MWR): Prioritize jobs with high remaining work to prevent bottlenecks.
    # 2. Remaining Ops Count: Prioritize jobs with fewer remaining steps to finish them off (completes jobs).
    # 3. Min Duration: In case of ties, prefer shorter operations to keep machines busy and allow more dispatches.
    
    # Precompute features
    rws = []
    min_durs = []
    rem_ops_counts = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        durations = list(mo.values()) if mo else []
        min_dur = min(durations) if durations else 0
        rem_ops = op.get("remaining_ops", 0)
        if isinstance(rem_ops, list):
            rem_ops_count = len(rem_ops)
        else:
            rem_ops_count = rem_ops
            
        rws.append(rw)
        min_durs.append(min_dur)
        rem_ops_counts.append(rem_ops_count)

    # Convert to numpy arrays for vectorized operations
    rws_arr = np.array(rws, dtype=float)
    min_durs_arr = np.array(min_durs, dtype=float)
    rem_ops_arr = np.array(rem_ops_counts, dtype=float)

    # Normalize features to [0, 1] range to combine them effectively
    # We want high RW to be good (high score), low Rem Ops to be good (high score), low Min Dur to be good (high score)
    
    rw_max = np.max(rws_arr) if n_ops > 1 else 1.0
    rw_min = np.min(rws_arr) if n_ops > 1 else 0.0
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    
    rem_max = np.max(rem_ops_arr) if n_ops > 1 else 1.0
    rem_min = np.min(rem_ops_arr) if n_ops > 1 else 0.0
    rem_range = rem_max - rem_min if rem_max > rem_min else 1.0
    
    dur_max = np.max(min_durs_arr) if n_ops > 1 else 1.0
    dur_min = np.min(min_durs_arr) if n_ops > 1 else 0.0
    dur_range = dur_max - dur_min if dur_max > dur_min else 1.0

    # Normalized scores (higher is better)
    # RW: Higher is better
    norm_rw = (rws_arr - rw_min) / rw_range
    
    # Rem Ops: Lower is better (so we invert)
    norm_rem = 1.0 - (rem_ops_arr - rem_min) / rem_range
    
    # Min Dur: Lower is better (so we invert)
    norm_dur = 1.0 - (min_durs_arr - dur_min) / dur_range

    # Weighted Sum
    # Based on the insight that MWR is dominant, but balancing with completion (rem ops) helps.
    # Weights: MWR=1.0, RemOps=0.3, MinDur=0.1
    # These are tuned to prioritize the critical path (MWR) while slightly favoring job completion and short ops.
    
    weights = np.array([1.0, 0.3, 0.1])
    
    scores = weights[0] * norm_rw + weights[1] * norm_rem + weights[2] * norm_dur

    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking logic: If there are ties within a small epsilon, use a secondary heuristic.
    # Secondary: Prefer the operation that allows the earliest completion time (min duration) on its best machine.
    # This is already partially captured in the score, but explicit tie-breaking can be more robust.
    max_score = scores[best_idx]
    eps = 1e-9
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < eps]
    
    if len(ties) > 1:
        # Break tie by preferring the operation with the smallest minimum duration
        # This keeps machines busy with quick tasks if work remaining is equal
        def secondary_score(idx):
            op = ready_operations[idx]
            durations = list(op["machine_options"].values())
            return min(durations) if durations else float('inf')
        
        # If still tied, prefer the one with fewer remaining ops
        def tertiary_score(idx):
            op = ready_operations[idx]
            rem_ops = op.get("remaining_ops", 0)
            if isinstance(rem_ops, list):
                return len(rem_ops)
            return rem_ops

        # Sort ties by secondary then tertiary
        ties_sorted = sorted(ties, key=lambda i: (secondary_score(i), tertiary_score(i)))
        best_idx = ties_sorted[0]
        
    return best_idx
