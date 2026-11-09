"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_LIMIT = 145.0


def _elapsed():
    return _clock.time() - _T0


def _duration_stats(opts):
    if not opts:
        return (0.0, 0.0, 0.0, 1)
    vs = list(opts.values())
    n = len(vs)
    if n == 1:
        return (vs[0], vs[0], vs[0], 1)
    M = max(vs)
    m = min(vs)
    if M == m:
        return (m, M, m, n)
    # Soft-max: M + log(sum(exp(v - M)))
    soft_max = M + math.log(sum(math.exp(v - M) for v in vs))
    # Soft-min: m - log(sum(exp(m - v)))
    soft_min = m - math.log(sum(math.exp(m - v) for v in vs))
    return (soft_min, soft_max, m, n)


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        rem = op["remaining_work"]
        soft_min, soft_max, min_dur, n_opts = _duration_stats(op["machine_options"])
        rem_ops = op.get("remaining_ops", 0)
        
        # Core heuristic: Longest Remaining Processing Time (LRPT)
        # This is generally the best static rule for open shop.
        # We add a small penalty for operations that are "slow" even on the fastest machine
        # to prefer faster operations if remaining work is similar.
        # We also add a tiny bonus for having more machine options (flexibility).
        
        # Weighting:
        # 1.0 * remaining_work: Primary driver (LRPT)
        # -0.1 * min_dur: Prefer faster ops if work is close
        # +0.05 * log(1 + n_opts): Small bonus for flexibility
        
        priority = rem - 0.1 * min_dur + 0.05 * math.log1p(n_opts)
        
        # Tie-breaker: lower index first (stable)
        return (priority, -idx)

    ranked = [(_priority(op, i), i) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
