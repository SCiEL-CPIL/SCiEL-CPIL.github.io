"""Guided local search penalty for TSP – topological disruption design."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    Strategy: topological disruption.
      1. Identify the k longest edges in the current tour (these are the
         structurally 'stuck' edges that prevent 2-opt from finding better
         solutions).
      2. Apply a heavy additive penalty to those edges to force 2-opt to
         break them.
      3. Add a mild log-scaled usage penalty on all tour edges to discourage
         recurring edges.
    """
    n = edge_distance.shape[0]
    result = edge_distance.copy()

    # Extract tour edges
    tour = local_opt_tour
    edges = []
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        if a != b:
            edges.append((a, b))

    if not edges:
        return result

    # Get lengths of tour edges
    edge_lengths = np.array([edge_distance[a, b] for a, b in edges])

    # Identify the longest edges (top 20% by length)
    k = max(1, len(edges) // 5)
    # Sort indices by length descending
    sorted_idx = np.argsort(edge_lengths)[::-1]
    top_k_idx = sorted_idx[:k]

    # Heavy additive penalty for the longest tour edges
    # Scale penalty relative to the median tour edge length
    median_len = np.median(edge_lengths)
    heavy_penalty = 2.0 * median_len  # large enough to force restructuring

    for idx in top_k_idx:
        a, b = edges[idx]
        result[a, b] += heavy_penalty
        result[b, a] += heavy_penalty

    # Mild log-scaled usage penalty on all tour edges
    # This discourages edges that have been used many times
    max_used = edge_n_used.max()
    if max_used > 0:
        for a, b in edges:
            usage = edge_n_used[a, b]
            if usage > 0:
                penalty = 0.05 * median_len * np.log1p(usage)
                result[a, b] += penalty
                result[b, a] += penalty

    # Ensure symmetry and finiteness
    result = np.maximum(result, 0.0)
    result = np.where(np.isfinite(result), result, edge_distance)

    return result
# EVOLVE-BLOCK-END
