"""Guided local search penalty rule for the TSP (3 fixed 100-point instances)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Wall-clock guard: the evaluator kills at 330 s. Keep a safety margin so we
# always return a valid finite matrix well before the hard limit.
_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 315.0  # seconds; margin below the 330 s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we are running out of time, return a safe, finite
    # matrix (the true distances) so the result stays valid.
    if (time.time() - _GLOBAL_START_TIME) > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Work in float64 to guarantee finite, well-conditioned output.
    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Guided local search: penalize the edges of the current local optimum,
    # scaled by how many times each has appeared in local optima (edge_n_used).
    # Use a smooth, bounded, monotone transform  u/(1+u)  (soft saturation) so
    # frequently-used edges get the strongest push without unbounded growth.
    penalty_strength = 1.0
    for i in range(n):
        a = int(local_opt_tour[i])
        b = int(local_opt_tour[(i + 1) % n])
        if a == b:
            continue
        u = U[a, b]
        p = penalty_strength * (u / (1.0 + u))
        D[a, b] = D[a, b] + p
        D[b, a] = D[b, a] + p

    # --- Small symmetric "spread" term on every edge to break near-degenerate
    # ties, add diversity, and keep the matrix strictly positive / finite.
    # Bounded by (1 - 1/(1+maxU)) < 1, so it never overwhelms the GLS penalty.
    max_u = float(U.max()) if U.size else 0.0
    spread = 0.05 * (max_u / (1.0 + max_u))
    D = D + spread

    # Ensure a finite, positive, same-shape matrix.
    D = np.nan_to_num(D, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    D = np.maximum(D, 0.0)
    return D
# EVOLVE-BLOCK-END
