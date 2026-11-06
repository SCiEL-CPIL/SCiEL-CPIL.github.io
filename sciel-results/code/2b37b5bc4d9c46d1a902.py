"""Job-Shop Scheduling dispatch rule with structural multi-factor heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = None
_DEADLINE = 140.0  # Safety margin before 150s kill


def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()

    # Deadline guard
    elapsed = _time.time() - _start_time
    if elapsed > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Pre-compute statistics
    max_remaining_work = 0.0
    max_remaining_ops = 0
    max_min_duration = 0.0
    max_avg_duration = 0.0
    min_min_duration = float('inf')

    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        durations = list(op.get("machine_options", {}).values())
        if durations:
            min_d = min(durations)
            avg_d = sum(durations) / len(durations)
        else:
            min_d = 0.0
            avg_d = 0.0

        if rw > max_remaining_work:
            max_remaining_work = rw
        if ro > max_remaining_ops:
            max_remaining_ops = ro
        if min_d > max_min_duration:
            max_min_duration = min_d
        if min_d < min_min_duration:
            min_min_duration = min_d
        if avg_d > max_avg_duration:
            max_avg_duration = avg_d

    # Avoid division by zero
    if max_remaining_work <= 0:
        max_remaining_work = 1.0
    if max_remaining_ops <= 0:
        max_remaining_ops = 1
    if max_min_duration <= 0:
        max_min_duration = 1.0
    if max_avg_duration <= 0:
        max_avg_duration = 1.0
    if min_min_duration == float('inf'):
        min_min_duration = 1.0

    # Compute score for each operation
    # Structural change: use a multiplicative penalty model combined with
    # a "critical path" estimate and machine contention awareness.
    #
    # Key insight: In job-shop scheduling, the makespan is dominated by the
    # critical path. We want to prioritize operations that are on or near
    # the critical path. A proxy for this is:
    #   1. Remaining work (total) - high means critical
    #   2. Remaining ops - fewer means closer to completion (urgency)
    #   3. Processing time of current op - longer ops are more critical
    #   4. Machine flexibility - ops with fewer machine options are more constrained
    #   5. Soft-min of durations (captures machine speed distribution)

    scores = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        durations = list(op.get("machine_options", {}).values())
        if durations:
            min_d = min(durations)
            avg_d = sum(durations) / len(durations)
            # Soft-min via log-sum-exp for smoother surrogate
            beta = 0.1  # temperature parameter
            max_d = max(durations)
            log_sum_exp = max_d + math.log(sum(math.exp(-beta * (d - max_d)) for d in durations))
            soft_min = max_d - log_sum_exp / beta
            # Actually, for soft-min we want: -1/beta * log(sum(exp(-beta*d)))
            # Let me redo: soft_min = -1/beta * log(sum(exp(-beta*d)))
            neg_beta_d = [-beta * d for d in durations]
            max_neg = max(neg_beta_d)
            log_sum = max_neg + math.log(sum(math.exp(x - max_neg) for x in neg_beta_d))
            soft_min = -log_sum / beta

            n_machines = len(durations)
        else:
            min_d = 0.0
            avg_d = 0.0
            soft_min = 0.0
            n_machines = 1

        # Component 1: Normalized remaining work (importance)
        mwr = rw / max_remaining_work

        # Component 2: Urgency - jobs with fewer remaining ops are closer to done
        # We want to prioritize them to free up machines
        if ro > 0:
            urgency = (max_remaining_ops - ro + 1) / (max_remaining_ops + 1)
        else:
            urgency = 1.0

        # Component 3: Current operation processing time (LPT-like)
        # Use soft-min to capture machine speed distribution
        lpt = soft_min / max_min_duration if max_min_duration > 0 else 0.0

        # Component 4: Machine constraint - ops with fewer machine options
        # are more constrained and should be prioritized
        # Normalize: 1/n_machines (higher when fewer options)
        constraint = 1.0 / n_machines if n_machines > 0 else 1.0

        # Component 5: Work density - remaining work per remaining op
        # High density means each op is heavy, so completing one is significant
        if ro > 0:
            density = rw / (ro * max_remaining_work)
            density = min(density, 1.0)
        else:
            density = 0.0

        # Component 6: Critical path estimate
        # Estimate how much this job contributes to the makespan
        # If a job has a lot of remaining work and few remaining ops,
        # its remaining ops are likely long and critical
        if ro > 0:
            avg_remaining_op = rw / ro
            # Normalize by the max possible average
            max_avg_op = max_remaining_work / max(1, min(1, max_remaining_ops))
            if max_avg_op > 0:
                crit_path = min(avg_remaining_op / max_avg_op, 1.0)
            else:
                crit_path = 0.0
        else:
            crit_path = 0.0

        # Composite score: weighted sum with emphasis on critical path factors
        # Weights tuned to prioritize:
        # - Critical path / remaining work (0.30)
        # - Current op length (0.25)
        # - Urgency / proximity to completion (0.20)
        # - Machine constraint (0.10)
        # - Work density (0.10)
        # - MWR (0.05)

        score = (
            0.30 * crit_path +
            0.25 * lpt +
            0.20 * urgency +
            0.10 * constraint +
            0.10 * density +
            0.05 * mwr
        )

        scores.append(score)

    # Return index of the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
