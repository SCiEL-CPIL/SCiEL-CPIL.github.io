"""Rewritten program: dead-gap heuristic for ant colony bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    # Sort demands to allow efficient counting via searchsorted
    sorted_demands = np.sort(demands)

    # Calculate residual space for every pair (i, j)
    # residual[i, j] = capacity - (demands[i] + demands[j])
    residual = capacity - (demands[:, None] + demands[None, :])

    # Count how many items fit in the residual space for each pair.
    # searchsorted(sorted_demands, residual, side='right') returns the index
    # where the residual value would be inserted to keep order.
    # Since sorted_demands is sorted, this index equals the count of elements
    # in sorted_demands that are <= residual.
    # This count represents the number of items k such that demands[k] <= residual[i, j].
    counts = np.searchsorted(sorted_demands, residual, side='right')

    # The heuristic is inversely proportional to the count of fitting items.
    # count = 0 (dead gap) -> 1.0
    # count = 1 -> 0.5
    # count = k -> 1/(k+1)
    heuristic = 1.0 / (counts + 1.0)

    # Pairs that do not fit together (residual < 0) are invalid and should have
    # minimal attractiveness.
    heuristic = np.where(residual >= 0, heuristic, 1e-9)

    # Safety checks: ensure positive and finite values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
