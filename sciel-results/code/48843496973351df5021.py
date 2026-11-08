"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global start time for deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _pick_machine(op):
    """Choose the machine with shortest processing time for this operation."""
    machine_options = op["machine_options"]
    if not machine_options:
        return None
    best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
    return best_machine


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Uses a priority-weighted combination of Remaining Work, Shortest Processing Time,
    and Remaining Operations to balance job completion urgency with machine utilization.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we are running low on time, fall back to a very fast heuristic
    if _time.time() - _START_TIME > _DEADLINE:
        # Fallback: Most work remaining first (fast)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        idx = 0
        m = _pick_machine(ready_operations[0])
        if m is not None:
            return (idx, m)
        return idx

    # Gather features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    spt = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    def _normalize(a):
        """Normalize to [0, 1] range; higher is better for the objective."""
        rng = a.max() - a.min()
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - a.min()) / rng

    norm_work = _normalize(remaining_works)
    norm_spt = _normalize(spt)
    norm_ops = _normalize(remaining_ops)

    # Priority-weighted score: higher is better
    # Combine:
    #   - High remaining work (MWRF): prioritize jobs with more total work left
    #   - Low processing time (SPT): prioritize operations that finish quickly
    #   - High remaining ops: prioritize jobs with more operations remaining
    #   - Ratio (work/spt): balances urgency with immediate completion
    W_WORK = 0.4
    W_SPT = 0.3
    W_OPS = 0.2
    W_RATIO = 0.1

    # Compute ratio feature: work / (spt + epsilon) to avoid division by zero
    ratio = remaining_works / (spt + 1e-9)
    norm_ratio = _normalize(ratio)

    scores = (
        W_WORK * norm_work +
        W_SPT * (1.0 - norm_spt) +
        W_OPS * norm_ops +
        W_RATIO * norm_ratio
    )

    idx = int(np.argmax(scores))

    # Choose machine: shortest processing time
    m = _pick_machine(ready_operations[idx])
    if m is not None:
        return (idx, m)
    return idx
# EVOLVE-BLOCK-END
