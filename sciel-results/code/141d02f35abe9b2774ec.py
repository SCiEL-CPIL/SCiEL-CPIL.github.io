"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE_START = _time.time()
_MAX_WALL = 145.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Primary signal: longest remaining work (LRPT) - proven effective
    # Secondary signal: most remaining operations (tie-breaker)
    # This combines the structural simplicity of the inspiration with
    # a lightweight refinement from the current program's feature set.

    best_idx = 0
    best_work = ready_operations[0]["remaining_work"]
    best_ops = ready_operations[0]["remaining_ops"]

    for i in range(1, n):
        work = ready_operations[i]["remaining_work"]
        ops = ready_operations[i]["remaining_ops"]

        # Compare with small tolerance to avoid floating-point issues
        if work > best_work + 1e-9:
            best_idx = i
            best_work = work
            best_ops = ops
        elif abs(work - best_work) <= 1e-9:
            # Tie-break: prefer more remaining operations
            if ops > best_ops:
                best_idx = i
                best_ops = ops

    return best_idx
# EVOLVE-BLOCK-END
