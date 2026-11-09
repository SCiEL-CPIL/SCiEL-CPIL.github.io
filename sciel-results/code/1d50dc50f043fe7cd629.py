"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1).astype(float)
    # Sum of neighbor degrees
    neighbor_deg_sum = adjacency @ degree
    # Score: prefer low degree, but boost vertices whose neighbors are high-degree
    # (picking them "saves" high-degree neighbors from being wasted)
    # Normalize neighbor degree contribution
    mean_neighbor_deg = neighbor_deg_sum / np.maximum(degree, 1.0)
    # Combine: negative degree (primary), plus bonus for high neighbor degrees
    score = -degree + 0.3 * mean_neighbor_deg
    return score
# EVOLVE-BLOCK-END
