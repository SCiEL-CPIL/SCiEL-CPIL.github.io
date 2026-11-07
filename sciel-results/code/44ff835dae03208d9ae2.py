"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    # Favor pairs that nearly fill the bin (high utilization)
    # but also reward smaller items for flexibility
    # Use a combination: utilization * (1 + small bonus for smaller j)
    util = np.where(pair <= capacity, pair / capacity, 0.0)
    # Smaller items are more flexible - add a mild bias
    # inv_demands[j] is larger for smaller j
    inv_d = 1.0 / demands
    # Blend: 70% utilization, 30% flexibility (favoring smaller j)
    eta = 0.7 * util + 0.3 * inv_d[None, :]
    # Ensure non-positive entries become 1e-9
    eta = np.where(eta > 0, eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
