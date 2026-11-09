"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_LIMIT = 135.0


def _elapsed():
    return _clock.time() - _T0


def _logsumexp_soft_min(opts):
    """Smooth min via log-sum-exp (negative trick)."""
    if not opts:
        return 0.0
    vals = list(opts.values())
    if len(vals) == 1:
        return vals[0]
    # soft min = -logsumexp(-x_i)
    negs = [-v for v in vals]
    m = max(negs)
    if m == float('-inf'):
        return 0.0
    s = sum(math.exp(n - m) for n in negs)
    return -(m + math.log(s))


def _logsumexp_soft_max(opts):
    """Smooth max via log-sum-exp."""
    if not opts:
        return 0.0
    vals = list(opts.values())
    if len(vals) == 1:
        return vals[0]
    m = max(vals)
    if m == float('-inf'):
        return 0.0
    s = sum(math.exp(v - m) for v in vals)
    return m + math.log(s)


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        rem = op["remaining_work"]
        opts = op["machine_options"]
        
        # Smooth min and max using log-sum-exp
        s_min = _logsumexp_soft_min(opts)
        s_max = _logsumexp_soft_max(opts)
        
        # Flexibility: number of machine options
        flexibility = len(opts) if opts else 0
        
        # Remaining ops count
        rem_ops = op.get("remaining_ops", 0)
        
        # Composite score:
        # 1. Maximize remaining work (LRPT) - primary
        # 2. Minimize soft-min duration (prefer ops that can be done fast somewhere)
        # 3. Maximize flexibility (more machine options = less bottlenecks)
        # 4. Maximize soft-max (long tasks shouldn't be starved)
        # 5. Minimize remaining ops (finish jobs that are almost done)
        # 6. Tie-break by index
        score = (
            rem,              # maximize
            -s_min,           # minimize soft-min (i.e., maximize -s_min)
            flexibility,      # maximize
            s_max,            # maximize
            -rem_ops,         # minimize
            -idx              # tie-break
        )
        return score

    ranked = [(_priority(op, i), i) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
