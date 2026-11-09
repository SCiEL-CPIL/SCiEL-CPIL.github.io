"""Improved job-shop dispatch rule: combine machine load, remaining work, and critical path."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; use 140s as safety margin
_DEADLINE = _time.time() + 140.0


def _compute_score(ready_operations, time):
    """Compute a composite score for each ready operation. Higher = better to pick."""
    n = len(ready_operations)
    if n == 0:
        return 0

    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        # 1. Remaining work (total processing time left for this job)
        remaining_work = op.get("remaining_work", 0)

        # 2. Remaining operations count (jobs with more ops left are more urgent)
        remaining_ops = op.get("remaining_ops", 0)

        # 3. Minimum duration across machine options (shorter ops first to reduce blocking)
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_dur = min(durations)
            mean_dur = sum(durations) / len(durations)
        else:
            min_dur = 0
            mean_dur = 0

        # 4. Number of machine options (more flexibility = less urgent)
        n_opts = len(machine_options) if machine_options else 1

        # Composite score:
        # - Higher remaining work -> more urgent (prevents long jobs from stalling)
        # - Higher remaining ops -> more urgent (jobs close to completion should finish)
        # - Lower min duration -> slightly preferred (reduces machine blocking)
        # - Fewer machine options -> more urgent (less flexible jobs should go first)

        # Normalize-like weighting (no global normalization needed since we just rank)
        score = 0.0
        score += 1.0 * remaining_work          # primary: total work remaining
        score += 0.5 * remaining_ops          # secondary: number of ops left
        score -= 0.3 * min_dur                # slight preference for shorter ops
        score += 0.2 * (1.0 / max(n_opts, 1)) # urgency for inflexible ops

        scores[i] = score

    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() > _DEADLINE - 5.0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
