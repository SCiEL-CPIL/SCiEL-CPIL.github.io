"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # (5, 100) -> (100,)
    total_prize = prizes.sum(axis=0)

    # Per-constraint efficiency: prize / weight for each constraint
    # (5, 100)
    eff = prizes / (weights + 1e-9)

    # Soft-min via log-sum-exp: smooth surrogate for the minimum constraint efficiency.
    # As alpha -> inf, this approaches min(eff, axis=0); for moderate alpha it
    # penalizes imbalance without being dominated entirely by the single worst constraint.
    alpha = 10.0
    log_eff = np.log(eff + 1e-9)
    # Soft-min: -logsumexp(-alpha * log_eff) / alpha  ~  exp(-logsumexp(-alpha*log_eff)/alpha)
    # Equivalent to: exp( (-logsumexp(-alpha * log_eff)) / alpha )
    # But more stably: use log-space throughout
    # soft_min_log = -logsumexp(-alpha * log_eff, axis=0) / alpha
    # We compute in log-space to avoid overflow
    neg_alpha_log = -alpha * log_eff  # (5, 100)
    # logsumexp along axis 0
    max_neg = neg_alpha_log.max(axis=0, keepdims=True)
    logsumexp_neg = max_neg.squeeze(0) + np.log(np.exp(neg_alpha_log - max_neg).sum(axis=0))
    soft_min_log = -logsumexp_neg / alpha  # log of soft-min efficiency
    soft_min = np.exp(soft_min_log)  # (100,)

    # Arithmetic mean of per-constraint efficiencies (more lenient than geo mean)
    arith_mean = eff.mean(axis=0)  # (100,)

    # Sum-ratio: total prize / total weight across all constraints
    # (100,)
    total_weight = weights.sum(axis=0)
    sum_ratio = total_prize / (total_weight + 1e-9)

    # Blend: combine soft-min (bottleneck-aware), arith-mean (overall), and sum-ratio
    # Use geometric mean of the three signals for a balanced multiplicative blend
    # This rewards items that are efficient in the worst constraint AND overall
    log_blend = (soft_min_log + np.log(arith_mean + 1e-9) + np.log(sum_ratio + 1e-9)) / 3.0
    blend = np.exp(log_blend)

    # Ensure positive and finite
    blend = np.where(np.isfinite(blend) & (blend > 0), blend, 1e-9)

    # Check deadline (unlikely to be hit for simple arithmetic, but safe)
    if time.time() > deadline:
        return np.maximum(blend, 1e-9)

    return blend
# EVOLVE-BLOCK-END
