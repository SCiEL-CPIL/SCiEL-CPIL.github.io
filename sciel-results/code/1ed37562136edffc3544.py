"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Returns a priority score for each vertex.
    Higher score means higher priority (selected first in greedy decoder).

    Strategy (same family as before, refined):
    - Prefer vertices that are adjacent to high-degree hubs (so selecting them removes hubs).
    - Penalize vertices that sit inside dense local clusters (using a smooth surrogate for local edge density).
    - Penalize high own degree (since selecting a high-degree vertex blocks many others).

    Score components:
      1) avg_neighbor_degree: average degree of neighbors (high = good for hub-blocking)
      2) local_density_penalty: smooth penalty based on neighbor degree distribution
         (approximates local edge density without explicit O(N^3) work)
      3) own_degree_penalty: linear penalty for high degree

    We combine these into a single scalar score.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # leave ~5s safety margin under 120s

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]

    # Basic degree vector
    degrees = A.sum(axis=1)

    # Sum of neighbor degrees
    neighbor_degree_sums = A @ degrees

    # Average neighbor degree (0 for isolated vertices)
    avg_neighbor_degrees = np.zeros_like(degrees)
    mask = degrees > 0
    avg_neighbor_degrees[mask] = neighbor_degree_sums[mask] / degrees[mask]

    # Smooth local density penalty:
    # For each vertex i, consider the multiset of neighbor degrees {d_j}.
    # A cheap smooth surrogate for “local density” is the log-sum-exp of neighbor degrees,
    # normalized by degree, which increases when neighbors are collectively very high-degree.
    # We use a temperature to make it smooth and numerically stable.
    #
    # To avoid O(N^2) memory/time beyond matrix-vector products, we approximate:
    #   soft_density[i] = log( mean_j exp(d_j / T) ) over neighbors j
    # This can be computed as:
    #   log( (1/deg_i) * sum_j exp(d_j / T) )
    # We can compute sum_j exp(d_j / T) via A @ exp(degrees / T).
    T = 1.0  # temperature; tune if needed
    exp_neighbor = np.exp(degrees / T)
    sum_exp = A @ exp_neighbor
    # soft_density[i] = log(sum_exp[i] / deg_i) for deg_i > 0; 0 otherwise
    soft_density = np.zeros_like(degrees)
    soft_density[mask] = np.log(np.maximum(sum_exp[mask] / degrees[mask], 1e-12))

    # Normalize components to comparable scales
    # Use robust normalization to avoid huge dynamic ranges
    def _robust_normalize(x):
        if x.size == 0:
            return x
        # clip to avoid extreme outliers
        x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
        rng = x.max() - x.min()
        if rng <= 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    # Higher avg_neighbor_degree is good.
    # Higher soft_density is bad (dense local cluster).
    # Higher own degree is bad.
    a = _robust_normalize(avg_neighbor_degrees)
    b = _robust_normalize(soft_density)
    c = _robust_normalize(degrees)

    # Weights: emphasize hub-blocking, moderate penalty for local density, and own degree.
    w_a = 1.0
    w_b = 0.6
    w_c = 0.8

    scores = w_a * a - w_b * b - w_c * c

    # Optional: add a tiny bonus for isolated vertices (degree 0) since they never block others.
    # This keeps the family but helps break ties in favor of harmless picks.
    iso_bonus = 0.05 * (degrees == 0).astype(float)
    scores = scores + iso_bonus

    # Deadline guard: if we somehow exceed budget, return a simple fallback.
    if time.time() > deadline:
        return (-degrees).astype(float)

    return scores.astype(float)
# EVOLVE-BLOCK-END
