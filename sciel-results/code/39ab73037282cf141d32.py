"""Open-shop dispatch rule: composite urgency with deadline guard."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track start time for deadline guard
_START_TIME = _time.time()
_DEADLINE_S = 140.0  # safety margin before 150s kill


def _elapsed_ok():
    return (_time.time() - _START_TIME) < _DEADLINE_S


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.
    
    Uses a composite urgency score that balances:
    - remaining_work (total work left for the job)
    - shortest machine duration for this op (tightness)
    - work density (remaining_work / num remaining ops)
    - flexibility (number of machine options - prefer less flexible)
    - remaining_ops count (jobs with fewer ops left are more urgent)
    
    Returns index of the chosen operation.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we're running long, fall back to simple rule
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    scores = np.zeros(n, dtype=np.float64)
    min_durs = np.zeros(n, dtype=np.float64)
    flexibilities = np.zeros(n, dtype=np.float64)
    densities = np.zeros(n, dtype=np.float64)
    rem_ops = np.zeros(n, dtype=np.float64)
    rem_work = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        durs = list(mo.values())
        min_dur = min(durs) if durs else 1.0
        min_durs[i] = min_dur
        flexibilities[i] = float(len(durs))
        rw = float(op["remaining_work"])
        ro = float(op["remaining_ops"])
        rem_work[i] = rw
        rem_ops[i] = ro
        densities[i] = rw / max(ro, 1.0)

    # Normalize each component to [0, 1] range for consistent weighting
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # We want to maximize urgency:
    # - Higher remaining_work is more urgent
    # - Shorter min_dur means tighter constraint (urgency) -> invert
    # - Higher density means more critical
    # - Lower flexibility means more constrained -> invert
    # - Fewer remaining_ops means more urgent -> invert

    nw = normalize(rem_work)
    nd = normalize(min_durs)  # lower is better, so we invert
    nd_inv = 1.0 - nd
    ndens = normalize(densities)
    nflex = normalize(flexibilities)
    nflex_inv = 1.0 - nflex
    nops = normalize(rem_ops)
    nops_inv = 1.0 - nops

    # Composite score with tuned weights
    # Emphasize remaining work and tightness, with secondary signals
    w_work = 0.30
    w_tight = 0.25
    w_density = 0.15
    w_flex = 0.15
    w_ops = 0.15

    scores = (w_work * nw + w_tight * nd_inv + w_density * ndens +
              w_flex * nflex_inv + w_ops * nops_inv)

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
