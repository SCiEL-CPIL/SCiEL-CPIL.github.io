"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right
    after item i in the same bin.

    Strategy: Use a high-power fill ratio to sharply prefer pairs that nearly
    fill the bin, plus a smooth tight-fit bonus that rewards near-perfect
    complements with a Gaussian-like decay in waste. This guides ants to close
    bins with minimal leftover space while keeping the heuristic smooth.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)

    # Pair sums
    pair = demands[:, None] + demands[None, :]

    # Valid mask: pair must fit in the bin
    valid = pair <= capacity

    # Fill ratio: how much of the capacity the pair uses
    fill_ratio = np.where(valid, pair / capacity, 0.0)

    # High power exponent sharply rewards high fill ratios
    # Tuned exponent: 25 gives sharper discrimination between 0.9 and 1.0
    # 0.9^25≈0.0718, 0.95^25≈0.277, 0.98^25≈0.603, 1.0^25=1.0
    fill_power = fill_ratio ** 25

    # Smooth tight-fit bonus: Gaussian decay in waste
    # waste = capacity - pair_sum, for valid pairs
    waste = np.where(valid, capacity - pair, 0.0)
    # exp(-waste^2 / (2 * sigma^2)) with sigma=3.0
    # waste=0 -> 1.0, waste=1 -> 0.94, waste=3 -> 0.61, waste=5 -> 0.22, waste=10 -> 0.006
    sigma = 3.0
    tight_bonus = np.where(valid, np.exp(-(waste ** 2) / (2.0 * sigma ** 2)), 0.0)
    # Scale the bonus to be competitive with fill_power
    tight_bonus *= 1.5

    # Combine: primary fill signal + tight-fit bonus
    heuristic = np.where(valid, fill_power + tight_bonus, 0.0)

    # Ensure all entries are positive (minimum 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
