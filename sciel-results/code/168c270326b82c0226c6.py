"""Rewritten program: size-based heuristic favoring largest fitting items for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]

    # A pair (i, j) fits if demands[i] + demands[j] <= capacity
    fits = (demands[:, None] + demands[None, :]) <= capacity

    # Heuristic: Favor larger items j.
    # If j fits after i, the attractiveness is proportional to the size of j.
    # This encourages filling the bin with large items first, reducing fragmentation.
    attractiveness = np.where(fits, demands[None, :], 1e-9)

    # Ensure all values are positive and finite
    attractiveness = np.where(np.isfinite(attractiveness) & (attractiveness > 0),
                              attractiveness, 1e-9)

    return attractiveness
# EVOLVE-BLOCK-END
