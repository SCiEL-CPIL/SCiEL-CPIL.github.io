"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    # Distance from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    if k == 1:
        return int(feas[0])

    # Dynamic regret-minimization: steer away from outlier-long edges
    # by comparing each candidate's edge cost to the average of the feasible set.
    # score[i] = d_cur[i] - alpha * (mean(d_cur) - d_cur[i])
    #          = d_cur[i] * (1 + alpha) - alpha * mean(d_cur)
    # This is equivalent to minimizing d_cur[i] since mean is constant,
    # BUT we add a second-order term that penalizes deviation from the mean
    # to maintain uniform segment lengths and prevent capacity exhaustion.
    
    # Alternative: use a smooth surrogate that combines nearest-neighbor
    # with a penalty for being an outlier in the feasible set's distance distribution.
    d_mean = d_cur.mean()
    d_std = d_cur.std()
    if d_std < 1e-12:
        d_std = 1.0
    
    # Regret term: how much worse is this edge compared to average?
    # Positive regret = this edge is longer than average (bad)
    regret = d_cur - d_mean
    
    # Smooth penalty: use a soft-min approach to penalize outliers
    # score = d_cur + lambda * max(0, regret)^2 / d_std
    # This is nonsmooth, so use a smooth approximation:
    # smooth_max(x, 0) ≈ x + softplus(-x) where softplus(t) = log(1 + exp(t))
    # But for simplicity and speed, use a quadratic penalty on positive regret
    lambda_regret = 0.3
    penalty = lambda_regret * np.maximum(0.0, regret) ** 2 / d_std
    
    # Also add a small capacity-aware term: prefer nodes with lower demand
    # when remaining capacity is tight
    d_feas = demands[feas]
    cap_tightness = 1.0 / (rest_capacity + 1.0)
    gamma_cap = 0.1 * cap_tightness
    cap_penalty = gamma_cap * d_feas / (d_feas.max() + 1e-12)
    
    scores = d_cur + penalty + cap_penalty
    
    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
