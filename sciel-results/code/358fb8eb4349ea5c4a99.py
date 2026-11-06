"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track start time and bail out to a safe (nearest-feasible or depot) choice
# if we approach the limit, leaving a safety margin.
_START_TIME = None
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # seconds


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _get_start_time())


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for Capacitated VRP.

    Uses a Clarke-Wright savings-inspired composite score:
      - distance from current node to candidate
      - savings term: d(current,depot) - d(current,cand) + d(cand,depot)
        (lower savings = better consolidation)
      - demand penalty to preserve capacity for future visits
    Returns the best customer, or the depot (0) to close the route.
    """
    # Deadline guard: if we're about to run out of time, fall back to a
    # fast, safe choice.
    if _time_remaining() < 0.5:
        # Fast fallback: nearest feasible customer, else depot.
        if len(unvisited_nodes) == 0:
            return 0
        feasible = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return 0

    # Filter to feasible customers (demand fits remaining capacity).
    feasible = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible) == 0:
        return 0

    # Composite scoring inspired by Clarke-Wright savings.
    # The savings of inserting candidate i after current (instead of returning to depot
    # and starting a new route from i) is:
    #   saving = d(current, depot) + d(depot, i) - d(current, i)
    # Maximizing saving is equivalent to minimizing:
    #   cost = d(current, i) - d(current, depot) - d(depot, i)
    # But since d(current, depot) is constant for all candidates, we can simplify to:
    #   cost = d(current, i) - d(depot, i)
    # However, this can be negative. A more stable formulation that captures
    # the spirit of savings while remaining well-behaved:
    #   score = d(current, i) + w * d(i, depot) - w * d(current, depot)
    # The last term is constant, so effectively:
    #   score = d(current, i) + w * d(i, depot)
    # with w < 1 to capture the savings effect (reducing the weight of return trip).

    d_cur = distance_matrix[current_node, feasible]
    d_dep = distance_matrix[feasible, depot]
    d_cur_dep = distance_matrix[current_node, depot]
    d_dem = demands[feasible].astype(float)

    # Savings-inspired weight: w_return < 1 means we discount the return cost
    # because visiting this node saves us the round-trip to depot.
    # w_return = 0.5 gives a moderate savings effect.
    w_return = 0.5
    w_demand = 0.1

    # Scale demand penalty by 1/rest_capacity so it's a fraction of remaining cap.
    if rest_capacity > 0:
        demand_frac = d_dem / rest_capacity
    else:
        demand_frac = np.zeros_like(d_dem)

    # Clarke-Wright savings cost:
    #   score = d_cur + (w_return - 1) * d_dep + w_demand * demand_frac
    # With w_return = 0.5:
    #   score = d_cur - 0.5 * d_dep + 0.1 * demand_frac
    # This rewards nodes with good savings potential while preserving capacity.

    score = d_cur + (w_return - 1.0) * d_dep + w_demand * demand_frac

    # 2-step lookahead with soft-min for smoothness and robustness.
    n_feas = len(feasible)
    if n_feas > 1:
        # distance from each candidate to all other feasible candidates
        d_cand_cand = distance_matrix[feasible][:, feasible]
        # distance from each candidate back to depot
        d_cand_dep = distance_matrix[feasible, depot]
        
        # For each candidate i, we want a smooth estimate of the best next hop.
        # Instead of hard min, use soft-min (log-sum-exp based) for differentiability
        # and to avoid sensitivity to noise in the distance matrix.
        # soft_min(x, tau) = -tau * log(sum(exp(-x/tau)))
        
        # Exclude self by setting diagonal to large value
        mask = np.eye(n_feas, dtype=bool)
        d_cand_cand_masked = np.where(mask, 1e9, d_cand_cand)
        
        # Soft-min over next customers
        tau = 0.5
        # Shift for numerical stability
        min_cand = d_cand_cand_masked.min(axis=1)
        exp_terms = np.exp(-(d_cand_cand_masked - min_cand[:, None]) / tau)
        soft_min_cand = -tau * np.log(exp_terms.sum(axis=1)) + min_cand
        
        # Soft-min over depot return (with penalty)
        # We treat depot return as a single option, so soft-min is just the value itself,
        # but we can blend it smoothly with the customer options.
        # Let's combine them: soft_min of [min over customers, depot_return + penalty]
        depot_penalty = 0.3
        depot_option = d_cand_dep + depot_penalty
        
        # Combine soft_min_cand and depot_option using a smooth minimum
        # soft_min(a, b, tau) = -tau * log(exp(-a/tau) + exp(-b/tau))
        a = soft_min_cand
        b = depot_option
        min_ab = np.minimum(a, b)
        exp_a = np.exp(-(a - min_ab) / tau)
        exp_b = np.exp(-(b - min_ab) / tau)
        soft_min_next = -tau * np.log(exp_a + exp_b) + min_ab
        
        # Weight the lookahead
        w_lookahead = 0.3
        score += w_lookahead * soft_min_next
    else:
        # Only one feasible candidate; no lookahead needed.
        pass

    # Use soft-argmin for selection to reduce sensitivity to small score differences.
    # This acts as a smoothing of the argmin operation.
    temp_select = 0.5
    min_score = np.min(score)
    weights = np.exp(-(score - min_score) / temp_select)
    weights /= np.sum(weights)
    
    # If one candidate is strongly preferred, pick it.
    # Otherwise, pick the argmin of the original score (deterministic) 
    # but the weighting could be used for stochastic selection if desired.
    # For now, we stick to argmin of the smoothed score to maintain determinism
    # but the score itself is smoother due to the soft-min lookahead.
    best_idx = int(np.argmin(score))

    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
