"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(durs, temperature=0.3):
    """Smooth surrogate for min using log-sum-exp."""
    d = np.asarray(durs, dtype=np.float64)
    if len(d) == 1:
        return float(d[0])
    d_min = d.min()
    scaled = -(d - d_min) / temperature
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return d_min - temperature * log_sum


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path for single op
    if n == 1:
        return 0

    # Extract key metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min duration and avg duration across machine options
    best_durs = np.empty(n, dtype=np.float64)
    avg_durs = np.empty(n, dtype=np.float64)

    # Count how many ready ops compete for each machine (load-balancing signal)
    machine_competition = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_competition[m] = machine_competition.get(m, 0) + 1

    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()), dtype=np.float64)
        best_durs[i] = _soft_min(durs, temperature=0.3)
        avg_durs[i] = np.mean(durs)

    # Normalization to [0, 1] (higher = more urgent / more efficient)
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Urgency: more remaining work and more remaining ops => more urgent
    urgency_work = _normalize(remaining_works)
    urgency_ops = _normalize(remaining_ops)
    urgency = 0.7 * urgency_work + 0.3 * urgency_ops

    # Efficiency: shorter operations reduce idle time; invert so shorter = higher
    eff_best = _normalize(-best_durs)  # shorter soft-min duration -> higher
    eff_avg = _normalize(-avg_durs)
    efficiency = 0.6 * eff_best + 0.4 * eff_avg

    # Look-ahead bottleneck: score each op by the minimum completion time it could
    # achieve if assigned to its least-loaded eligible machine. This captures the
    # immediate machine contention impact rather than just static job attributes.
    bottleneck_scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        # For each eligible machine, estimate completion time as:
        # current time + duration + penalty for competing ops on that machine
        best_completion = float('inf')
        for m, dur in op["machine_options"].items():
            # Penalty scales with number of competing ready ops on this machine
            # and the average duration of those competitors (proxy for queue length)
            competitors = machine_competition.get(m, 1)
            # Simple proxy: assume each competitor adds a fraction of average duration
            avg_comp_dur = avg_durs.mean() if n > 0 else dur
            est_completion = time + dur + (competitors - 1) * 0.3 * avg_comp_dur
            if est_completion < best_completion:
                best_completion = est_completion
        bottleneck_scores[i] = best_completion

    # Invert: earlier achievable completion => higher priority
    load_balance = _normalize(-bottleneck_scores)

    # Combined score: urgency dominant, efficiency secondary, look-ahead tertiary
    scores = 1.0 * urgency + 0.4 * efficiency + 0.2 * load_balance

    # Pick the operation with the highest combined score
    best_idx = int(np.argmax(scores))

    return best_idx
# EVOLVE-BLOCK-END
