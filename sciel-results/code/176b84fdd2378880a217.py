"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fill_ratio = pair / capacity

    # For fitting pairs, reward closeness to capacity with a steeper power
    # and a smooth penalty for very small pairs (which waste bin space).
    # Use fill_ratio^3 to strongly favor tight fits, and multiply by a
    # smooth factor that dampens very low fill ratios (below ~0.4) to
    # discourage leaving large gaps.
    #
    # The damping factor is a smooth sigmoid-like function:
    #   damp = 1 - 0.5 * exp(-((fill_ratio - 0.4) / 0.15)^2)
    # This is ~0.5 when fill_ratio << 0.4 and ~1.0 when fill_ratio >= 0.4.

    fit_mask = pair <= capacity
    fr = np.where(fit_mask, fill_ratio, 0.0)

    # Base reward: cubic to strongly prefer near-full pairs
    base = fr ** 3

    # Smooth damping for very small pairs (waste penalty)
    damp = 1.0 - 0.5 * np.exp(-((fr - 0.4) / 0.15) ** 2)

    heuristic = np.where(fit_mask, base * damp + 0.01, 1e-9)

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
