"""Heuristic matrix for ACO-TSP: sharper power-law heuristic + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0


def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix (larger = more attractive).

    Uses a sharper power-law heuristic (1 / d^2) to strongly emphasize
    short edges, which guides ACO ants toward locally-optimal (short)
    moves. This compounds with pheromone: better early tours deposit
    more pheromone on good edges, improving convergence.

    A deadline guard ensures we return a valid result before the
    330s wall-clock limit.
    """
    # Deadline guard: if we're too close to the limit, return a safe
    # (but valid) heuristic immediately.
    if time.time() > _DEADLINE - 5.0:
        return 1.0 / (distance_matrix + 1e-9)

    d = np.asarray(distance_matrix, dtype=np.float64)
    # Sharper power law: 1 / d^2 emphasizes short edges more strongly
    # than 1 / d. The +1e-9 prevents division by zero for diagonal (0)
    # entries; non-positive values are clamped to 1e-9 by the evaluator.
    heuristic = 1.0 / (d + 1e-9) ** 2

    # Ensure all values are finite and positive (defensive).
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0),
                         heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
