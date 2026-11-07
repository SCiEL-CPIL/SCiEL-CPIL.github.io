"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(tour, dist):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    improved = True
    # Limit iterations to avoid long runtimes
    max_iters = 50
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Check if reversing tour[i..j] improves the tour
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1 mod n])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1 mod n])
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                current = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new < current - 1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 300.0 - 30.0  # 270s safety margin

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    # Ensure valid input
    if d.shape[0] != d.shape[1]:
        raise ValueError("Distance matrix must be square")

    # Build a good initial tour using nearest neighbor + 2-opt
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        # Find nearest unvisited node
        best_node = min(unvisited, key=lambda j: d[current, j])
        tour.append(best_node)
        unvisited.remove(best_node)
        current = best_node

    # Apply 2-opt to improve
    tour = _two_opt(tour, d)

    # Compute tour length
    tour_length = sum(d[tour[i], tour[(i + 1) % n]] for i in range(n))

    # Create heuristic matrix based on inverse distance
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)

    # Base heuristic: inverse distance to the power of alpha
    alpha = 2.0
    eta = 1.0 / (d_safe ** alpha + 1e-30)

    # Boost edges that are part of the good 2-opt tour
    # These edges are known to be good, so increase their attractiveness
    boost_factor = 2.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        eta[u, v] *= boost_factor
        eta[v, u] *= boost_factor

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    # Check time
    if time.time() > deadline:
        return eta

    # Optionally, run a second round of refinement with different alpha
    # Try a sharper heuristic and blend
    alpha2 = 3.0
    eta2 = 1.0 / (d_safe ** alpha2 + 1e-30)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        eta2[u, v] *= boost_factor
        eta2[v, u] *= boost_factor
    eta2 = np.where(np.isfinite(eta2) & (eta2 > 0), eta2, 1e-9)

    # Blend the two heuristics
    eta_blend = 0.5 * eta + 0.5 * eta2
    eta_blend = np.where(np.isfinite(eta_blend) & (eta_blend > 0), eta_blend, 1e-9)

    return eta_blend
# EVOLVE-BLOCK-END
