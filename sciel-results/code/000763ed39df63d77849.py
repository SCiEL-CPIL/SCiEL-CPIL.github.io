"""Rewritten program: Savings-based (Clarke-Wright) heuristic for node selection."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a savings-based heuristic.
    Savings for appending j after i: dist(depot,i) + dist(depot,j) - dist(i,j).
    Prefer feasible customers with high savings; break ties toward nearer customers.
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)

    # If no unvisited nodes, or capacity is too small for any, close the route.
    if n_unvisited == 0:
        return 0

    # Feasibility mask: demand[j] <= rest_capacity
    feas = demands[unvisited] <= rest_capacity
    if not np.any(feas):
        return 0

    # Restrict to feasible candidates
    feas_idx = unvisited[feas]

    # Distances from current_node to each feasible candidate
    dist_cur = distance_matrix[current_node, feas_idx]

    # Savings: dist(depot, current) + dist(depot, j) - dist(current, j)
    # This is the reduction in total distance if we serve j after current
    # rather than having a separate trip from depot to j.
    dist_depot_cur = distance_matrix[depot, current_node]
    dist_depot_j = distance_matrix[depot, feas_idx]
    savings = dist_depot_cur + dist_depot_j - dist_cur

    # We want to maximize savings. But also, if we're near the depot or have
    # few remaining, closing might be good. However, the heuristic should
    # generally keep going if there's a feasible customer with positive savings.
    #
    # Combined score: primarily maximize savings, with a small tie-breaker
    # preferring nearer customers (lower dist_cur) to avoid long detours.
    # Normalize savings to [0,1] range for combination.
    max_sav = np.max(savings)
    min_sav = np.min(savings)
    if max_sav > min_sav:
        sav_norm = (savings - min_sav) / (max_sav - min_sav)
    else:
        sav_norm = np.zeros_like(savings)

    # Prefer high savings; tie-break with lower distance
    max_dist = np.max(dist_cur)
    min_dist = np.min(dist_cur)
    if max_dist > min_dist:
        dist_norm = (dist_cur - min_dist) / (max_dist - min_dist)
    else:
        dist_norm = np.zeros_like(dist_cur)

    # Weight: 0.9 * savings_norm + 0.1 * (1 - dist_norm)
    score = 0.9 * sav_norm + 0.1 * (1.0 - dist_norm)

    # Pick the best
    best_local = np.argmax(score)
    return int(feas_idx[best_local])
# EVOLVE-BLOCK-END
