"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Safety margin (seconds) before the 330 s wall-clock kill, so we always
# return a valid matrix even under unexpected slowness.
_DEADLINE_MARGIN_S = 5.0


def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.

    Strategy (Multiplicative Tightness-Size Heuristic):
      This heuristic strongly rewards pairs that are both:
      1. Large in total size (high pair_sum / capacity).
      2. Tight in fit (low remaining space, i.e., 1/(remaining + 1)).

      By using a multiplicative combination, we ensure that a pair is only
      highly attractive if it satisfies BOTH conditions. A tight fit of small
      items is less attractive than a tight fit of large items, and a large
      pair with a loose fit is less attractive than a large pair with a tight
      fit. This sharpens the gradient towards optimal bin packing solutions
      where bins are filled to capacity with large items.

      eta[i, j] = (1 / (remaining + 1)) * (pair_sum / capacity)  if pair fits
      eta[i, j] = 1e-9                                           otherwise
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    # Pair sums and fit mask
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity

    # Remaining space for fitting pairs (large sentinel elsewhere)
    remaining = np.where(fits, capacity - pair_sums, capacity)

    # Build the matrix, defaulting to the non-positive floor of 1e-9
    eta = np.full((n, n), 1e-9, dtype=np.float64)

    if np.any(fits):
        # Tightness factor: 1 / (remaining + 1)
        # Ranges from 1.0 (perfect fit) down to ~1/(capacity+1)
        tightness = 1.0 / (remaining[fits] + 1.0)
        
        # Size factor: pair_sum / capacity
        # Ranges from min_sum/capacity up to 1.0
        size_factor = pair_sums[fits] / capacity
        
        # Multiplicative combination
        # This creates a strong preference for pairs that are both large and tight
        eta[fits] = tightness * size_factor

    # Enforce the contract: any non-positive entry becomes 1e-9
    # Note: tightness and size_factor are always positive for fitting pairs,
    # so eta[fits] should be positive. But we check just in case.
    eta[eta <= 0.0] = 1e-9

    # Guard against non-finite values (defensive; should not occur)
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
        eta[eta <= 0.0] = 1e-9

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE_MARGIN_S:
        pass

    return eta
# EVOLVE-BLOCK-END
