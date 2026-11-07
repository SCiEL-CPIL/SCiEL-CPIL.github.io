"""Rewritten program: waste-complementarity heuristic for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]

    # Remaining space after placing item i
    remaining = capacity - demands[:, None]          # shape (n, 1)
    # Size of candidate successor j
    cand = demands[None, :]                          # shape (1, n)

    # A pair (i, j) fits if demands[i] + demands[j] <= capacity
    fits = (demands[:, None] + demands[None, :]) <= capacity

    # Waste after placing both: capacity - demands[i] - demands[j]
    waste = capacity - demands[:, None] - demands[None, :]

    # Complementarity: how well j fills the remaining space after i.
    # ratio = remaining_after_i / demands[j]  -> high when j nearly equals remaining (tight fit)
    # Guard division by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cand > 0, remaining / np.maximum(cand, 1e-12), 0.0)

    # Only valid where it fits and waste is non-negative
    valid = fits & (waste >= 0) & (cand > 0)

    # Base attractiveness: tight-fit ratio (high when j closely matches remaining space)
    base = np.where(valid, ratio, 0.0)

    # Bonus for very low waste (near-perfect fill)
    # waste is in [0, capacity]; map small waste to a bonus
    waste_norm = np.clip(waste / max(capacity, 1e-12), 0.0, 1.0)
    low_waste_bonus = np.where(valid, (1.0 - waste_norm) ** 2, 0.0)

    # Combine: primary signal is complementarity ratio, secondary is low-waste bonus
    attractiveness = base + 0.5 * low_waste_bonus

    # Where not valid (doesn't fit), set to a tiny positive value
    attractiveness = np.where(valid, attractiveness, 1e-9)

    # Ensure all values are positive and finite
    attractiveness = np.where(np.isfinite(attractiveness) & (attractiveness > 0),
                              attractiveness, 1e-9)

    return attractiveness
# EVOLVE-BLOCK-END
