"""Savings-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin before the 120s wall-clock limit
_DEADLINE = _start_time + _TIMEOUT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer, or return 0 (depot) to close the route.

    Heuristic: score each feasible candidate by the incremental route cost
    (go to the customer, then leave it toward the depot), adjusted to keep
    remaining load balanced so capacity is not stranded. Lower score = better.
    """
    # Deadline guard: close the route early if we are running out of time.
    if time.time() > _DEADLINE:
        return 0

    if not unvisited_nodes:
        return 0

    unvisited = np.asarray(unvisited_nodes, dtype=int)
    d = np.asarray(distance_matrix)
    dem = np.asarray(demands)

    # Distances from current node to each candidate.
    d_cur = d[current_node, unvisited]
    # Distance from each candidate back to the depot (return cost).
    d_ret = d[unvisited, depot]
    # Candidate demand.
    dmd = dem[unvisited]

    # Feasibility: candidate must fit in remaining capacity.
    feasible = dmd <= rest_capacity
    if not np.any(feasible):
        # No single customer fits -> close the route.
        return 0

    # Incremental route cost: arrive at customer + leave it toward depot.
    # Using the depot-return as a proxy keeps the route "tight" and avoids
    # long dangling tails.
    base_cost = d_cur + d_ret

    # Load balance: prefer moves that leave enough capacity for more customers.
    # Penalty grows when the candidate consumes most of the remaining capacity.
    frac = np.where(rest_capacity > 0, dmd / float(rest_capacity), 1.0)
    # Penalize over-committing capacity (stranding), encourage moderate loads.
    balance_penalty = 0.15 * frac * frac

    score = base_cost + balance_penalty
    score = np.where(feasible, score, np.inf)

    idx = int(np.argmin(score))
    return int(unvisited[idx])
# EVOLVE-BLOCK-END
