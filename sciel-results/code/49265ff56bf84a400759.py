"""Combined dispatch rule: weighted critical-path/SPT score + smooth soft-max lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline guard
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute min duration and also identify the "best" machine
    min_durations = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            min_durations[i] = 1.0
        else:
            # Find machine with shortest duration
            best_m = min(mo.items(), key=lambda x: x[1])
            min_durations[i] = best_m[1]

    # Normalize each feature to [0, 1]
    def _norm(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    nw = _norm(remaining_work)
    no = _norm(remaining_ops)
    nd = _norm(min_durations)

    # --- Smooth soft-max lookahead (inspiration 1, vectorized) ---
    # For candidate i, the "remaining horizon" is the soft-max of the critical-path
    # work of the OTHER ready ops. We compute soft_max_{j != i}(remaining_work[j])
    # via the log-sum-exp identity so we never run an O(n^2) Python loop.
    #
    # soft_max(S) = T * log( sum_{j in S} exp(rw_j / T) )
    # soft_max(all \ {i}) = T * log( (sum_all exp - exp_i) )
    # Use log-sum-exp with a shift for numerical stability.
    T = 2.0  # temperature: smaller -> sharper max; larger -> smoother mean-like
    scaled = remaining_work / T
    m = scaled.max()
    if m < 1e-12:
        # Degenerate: no remaining work to look ahead at, skip the term.
        lookahead = np.zeros(n, dtype=np.float64)
    else:
        shifted = np.exp(scaled - m)
        sum_all = shifted.sum()
        # sum_except_i = sum_all - shifted[i]; guard against underflow to 0
        sum_except = sum_all - shifted
        sum_except = np.maximum(sum_except, 1e-300)
        # soft_max of the rest, in original units
        lookahead = T * (np.log(sum_except) + m)

    # Normalize the lookahead so it is comparable to the [0,1] features
    if lookahead.max() - lookahead.min() > 1e-12:
        nl = (lookahead - lookahead.min()) / (lookahead.max() - lookahead.min())
    else:
        nl = np.full(n, 0.5, dtype=np.float64)

    # Weighted score: we want to MAXIMIZE this.
    # - Higher remaining_work -> more urgent (critical path)
    # - Higher remaining_ops -> keep in motion
    # - Lower min_duration -> SPT benefit, so use (1 - nd)
    # - LOWER remaining horizon (soft-max of others) -> better, so use (1 - nl)
    w_work = 0.50
    w_ops = 0.20
    w_spt = 0.30
    w_look = 0.10  # small refinement weight so it nudges, not overturns

    score = (w_work * nw + w_ops * no + w_spt * (1.0 - nd)
             - w_look * nl)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
