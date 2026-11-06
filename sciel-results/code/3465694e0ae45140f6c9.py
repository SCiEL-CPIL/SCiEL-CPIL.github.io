"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.perf_counter()
_DEADLINE_LIMIT = 110.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    if time_module.perf_counter() - _START_TIME > _DEADLINE_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # Smooth soft-min via log-sum-exp: softmin(d) = -log(sum(exp(-d/tau)))*tau
    # For small tau, approximates min; larger tau gives smoother surrogate.
    tau = 0.3
    soft_mins = np.empty(n, dtype=float)
    soft_maxs = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = np.array(list(op["machine_options"].values()), dtype=float)
        if durations.size > 0:
            # soft-min: -tau * log(sum(exp(-d/tau)))
            neg_d_over_tau = -durations / tau
            # numerically stable
            max_neg = neg_d_over_tau.max()
            soft_mins[i] = -tau * (max_neg + np.log(np.sum(np.exp(neg_d_over_tau - max_neg))))
            # soft-max: tau * log(sum(exp(d/tau)))
            d_over_tau = durations / tau
            max_pos = d_over_tau.max()
            soft_maxs[i] = tau * (max_pos + np.log(np.sum(np.exp(d_over_tau - max_pos))))
        else:
            soft_mins[i] = 0.0
            soft_maxs[i] = 0.0

    # Min-max normalize each feature to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    rw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    sm_norm = normalize(soft_mins)
    sx_norm = normalize(soft_maxs)

    # Weighted score: prioritize remaining work, then ops, then shorter durations
    # Shorter durations get higher priority (1 - normalized)
    # Increase work weight, reduce duration weights for stronger tail-completion focus
    w_work = 0.60
    w_ops = 0.20
    w_softmin = 0.10
    w_softmax = 0.10

    # Count machine options as a flexibility bonus
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    no_norm = normalize(n_options)

    score = (w_work * rw_norm +
             w_ops * ro_norm +
             w_softmin * (1.0 - sm_norm) +
             w_softmax * (1.0 - sx_norm) +
             0.0 * no_norm)  # flexibility term included but weighted 0 for now

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
