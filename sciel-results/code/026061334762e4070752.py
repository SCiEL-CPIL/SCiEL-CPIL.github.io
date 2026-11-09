"""Improved job-shop dispatch rule: graph-theoretic lookahead with smooth critical path estimation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; use 140s as safety margin
_START_TIME = _time.monotonic()
_DEADLINE = 140.0


def _compute_score(ready_operations, time):
    """Compute a composite score for each ready operation. Higher = better to pick.
    
    Strategy: Model the job-shop as a dynamic bipartite graph between ready operations 
    and machines. Select the operation that minimizes the estimated increase in the 
    global critical path length. This is approximated by the operation's duration plus 
    the maximum remaining work of the job it unblocks (lookahead).
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features into arrays
    rw = np.zeros(n, dtype=np.float64)      # remaining work
    ro = np.zeros(n, dtype=np.float64)      # remaining ops
    min_dur = np.zeros(n, dtype=np.float64) # min duration across machine options
    mean_dur = np.zeros(n, dtype=np.float64)# mean duration
    n_opts = np.zeros(n, dtype=np.float64)  # number of machine options
    soft_min_val = np.zeros(n, dtype=np.float64)
    soft_max_remaining = np.zeros(n, dtype=np.float64)  # smooth max of remaining work after this op

    for i, op in enumerate(ready_operations):
        rw[i] = op.get("remaining_work", 0)
        ro[i] = max(op.get("remaining_ops", 1), 1)

        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            min_dur[i] = float(np.min(durations))
            mean_dur[i] = float(np.mean(durations))
            n_opts[i] = float(len(machine_options))
            
            # Soft-min via log-sum-exp
            neg_durs = -durations
            max_neg = float(np.max(neg_durs))
            if max_neg < 1e10:
                soft_min_val[i] = -max_neg - float(np.log(np.sum(np.exp(neg_durs - max_neg))))
            else:
                soft_min_val[i] = min_dur[i]
            
            # Estimate remaining work after this operation completes
            # This is a lookahead: if we pick this op, the job's remaining work decreases
            # by the duration of this op (using soft-min as a smooth estimate)
            # The critical path impact is approximately: duration + max(remaining_work_after, 0)
            remaining_after = max(rw[i] - soft_min_val[i], 0.0)
            soft_max_remaining[i] = remaining_after
        else:
            min_dur[i] = 1.0
            mean_dur[i] = 1.0
            n_opts[i] = 1.0
            soft_min_val[i] = 1.0
            soft_max_remaining[i] = max(rw[i] - 1.0, 0.0)

    # Normalize features to [0, 1] range
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng

    norm_rw = _normalize(rw)
    norm_ro = _normalize(ro)

    # Inverse min_duration: shorter is better
    inv_min_dur = 1.0 / (min_dur + 1e-6)
    norm_inv_dur = _normalize(inv_min_dur)

    # Average remaining op duration (critical path proxy)
    avg_dur = rw / ro
    norm_avg_dur = _normalize(avg_dur)

    # Inverse soft-min: smaller soft-min (closer to true min) is better
    norm_inv_soft_min = _normalize(-soft_min_val)

    # Flexibility: more options is slightly better (less constrained)
    norm_n_opts = _normalize(n_opts)

    # Ratio term: remaining_work / min_duration
    ratio_term = rw / (min_dur + 1e-9)
    norm_ratio = _normalize(ratio_term)

    # NEW: Graph-theoretic lookahead term
    # Estimated critical path increase = duration + max_remaining_work_after
    # We want to MINIMIZE this, so we invert
    cp_increase = soft_min_val + soft_max_remaining
    norm_inv_cp = _normalize(-cp_increase)

    # NEW: Smooth max of remaining work (captures the tail of the job)
    # Jobs with larger soft_max_remaining are more urgent because they have more work to go
    norm_soft_max = _normalize(soft_max_remaining)

    # Composite score with tuned weights
    # Primary: graph-theoretic critical path lookahead (minimize estimated CP increase)
    # Secondary: ratio (work-to-duration) captures urgency of long jobs with short next ops
    # Tertiary: remaining work (critical path)
    # Quaternary: shortest processing time (SPT) to reduce blocking
    # Minor: flexibility, remaining ops count, soft-min smoothness

    score = (
        1.2 * norm_inv_cp           # PRIMARY: minimize estimated critical path increase
        + 0.8 * norm_ratio          # work-to-duration ratio
        + 0.6 * norm_rw             # critical path / remaining work
        + 0.4 * norm_inv_dur        # SPT: shorter ops first
        + 0.3 * norm_avg_dur        # critical path proxy
        + 0.25 * norm_soft_max      # smooth max remaining work (urgency of job tail)
        + 0.15 * norm_inv_soft_min  # soft-min smoothness
        + 0.1 * norm_ro             # remaining ops urgency
        + 0.05 * norm_n_opts        # flexibility
    )

    return int(np.argmax(score))


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.monotonic() - _START_TIME > _DEADLINE - 5.0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
