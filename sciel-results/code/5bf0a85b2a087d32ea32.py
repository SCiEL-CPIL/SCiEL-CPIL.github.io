"""Rewritten program: dead-gap heuristic for ant colony bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    # pair_sum[i, j] = d_i + d_j
    pair_sum = demands[:, None] + demands[None, :]

    # residual[i, j] = capacity - (d_i + d_j); negative means pair doesn't fit
    residual = capacity - pair_sum

    # For each pair, count how many items k fit into the residual space.
    # An item k fits if d_k <= residual[i, j].
    # We compute for all pairs at once: residual has shape (n, n), demands has shape (n,).
    # count[i, j] = number of k such that demands[k] <= residual[i, j]
    # Use broadcasting: residual[:, :, None] vs demands[None, None, :]
    # That would be (n, n, n) which is 120^3 = 1.7M — fine.
    fits = (residual[:, :, None] >= demands[None, None, :])  # shape (n, n, n), bool
    # Only count items that actually fit (residual >= 0 is implicit since demands >= 20)
    # But if residual < 0, no item fits anyway (demands >= 20 > 0), so fits will be all False.
    count = fits.sum(axis=2)  # shape (n, n)

    # Heuristic: reward pairs with few fitting items in residual (dead gaps).
    # If count == 0, it's a perfect dead gap -> highest attractiveness.
    # Use 1 / (1 + count**2) to strongly penalize gaps that can fit many items.
    # count=0 -> 1.0, count=1 -> 0.5, count=2 -> 0.2, count=10 -> 0.0099.
    # This sharpens the preference for "dead" spaces.
    heuristic = 1.0 / (1.0 + count**2)
    heuristic = np.where(residual >= 0, heuristic, 1e-9)

    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
