"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses a balanced multi-criteria heuristic tuned for job-shop scheduling:
    1. Prioritize jobs with more remaining work (reduces makespan by keeping pipeline full)
    2. Among similar work amounts, prefer shorter processing times (reduces idle time)
    3. Break ties by fewer remaining operations (completes jobs sooner)
    
    Returns the index of the chosen operation.
    """
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Collect features for all operations
    features = []
    for i, op in enumerate(ready_operations):
        options = op.get('machine_options', {})
        if not options:
            features.append((0, 0, 0))  # Low priority
            continue

        min_dur = min(options.values())
        remaining_work = op.get('remaining_work', 0)
        remaining_ops = op.get('remaining_ops', 0)
        features.append((remaining_work, min_dur, remaining_ops))

    # Find max remaining work for normalization
    max_work = max(f[0] for f in features) if features else 1
    min_min_dur = min(f[1] for f in features) if features else 1
    max_min_dur = max(f[1] for f in features) if features else 1
    max_remaining_ops = max(f[2] for f in features) if features else 1

    best_idx = 0
    best_score = float('-inf')

    for i, (rw, md, ro) in enumerate(features):
        if rw == 0 and md == 0 and ro == 0:
            continue

        # Normalize to [0, 1]
        norm_work = rw / (max_work + 1e-9) if max_work > 0 else 0
        # For duration: shorter is better, so invert
        norm_dur = 1.0 - (md - min_min_dur) / (max_min_dur - min_min_dur + 1e-9) if max_min_dur > min_min_dur else 1.0
        # For remaining ops: fewer is better
        norm_ops = 1.0 - ro / (max_remaining_ops + 1e-9) if max_remaining_ops > 0 else 1.0

        # Weighted score: emphasize remaining work most, then duration, then ops
        # Weights tuned for job-shop: work matters most for makespan
        score = 0.6 * norm_work + 0.3 * norm_dur + 0.1 * norm_ops

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
