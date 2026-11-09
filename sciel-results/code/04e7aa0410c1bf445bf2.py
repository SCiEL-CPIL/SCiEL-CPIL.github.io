"""Guided local search penalty rule for TSP with adaptive two-tier penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a two-tier penalty to guide the local search:
    - Base penalty proportional to edge usage frequency (smooth).
    - Additional boost for edges currently in the local-optimal tour.
    
    This makes the current tour edges significantly more expensive,
    forcing 2-opt to find alternative routes, while the base penalty
    discourages frequently-used edges overall.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Ensure we work with float arrays
    ed = np.asarray(edge_distance, dtype=np.float64)
    enu = np.asarray(edge_n_used, dtype=np.float64)

    # Safety: if max usage is 0, no penalty needed
    max_usage = enu.max()
    if max_usage == 0:
        return ed.copy()

    # Normalize usage: scale so that the most-used edge has penalty factor ~1.0
    # Use a smooth normalization: usage / (1 + max_usage) gives values in [0, ~0.5]
    # We want the penalty to be meaningful, so scale up
    norm_usage = enu / (1.0 + max_usage)

    # Base penalty: multiplicative factor (1 + alpha * norm_usage)
    # alpha controls how strongly we penalize frequently-used edges
    alpha_base = 0.5

    # Tour-edge boost: edges in the current tour get an extra penalty
    # This is the key GLS mechanism: make the current solution's edges expensive
    tour_edges = np.zeros_like(ed)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0

    # Boost factor for tour edges
    alpha_tour = 1.0

    # Combined penalty matrix
    penalty = 1.0 + alpha_base * norm_usage + alpha_tour * tour_edges

    # Apply penalty: multiply true distances by penalty
    # This makes penalized edges more expensive, so 2-opt will avoid them
    penalized = ed * penalty

    # Ensure finiteness
    if not np.all(np.isfinite(penalized)):
        return ed.copy()

    return penalized
# EVOLVE-BLOCK-END
