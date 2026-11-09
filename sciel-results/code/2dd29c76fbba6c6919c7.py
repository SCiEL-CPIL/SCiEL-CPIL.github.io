"""Fresh design: sparse, tour-edge-only penalty for Guided Local Search."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Perturbation step for Guided Local Search.

    Returns a finite distance matrix of the same shape as edge_distance.
    Only the edges of the current local-optimum tour are penalized; every
    other edge keeps its true distance. The penalty is scaled to the tour's
    mean edge length and grows with how many times each specific tour edge
    has been used (edge_n_used), so repeatedly-used edges are pushed hardest.
    """
    n = edge_distance.shape[0]
    tour = np.asarray(local_opt_tour, dtype=np.int64)

    # Safety: fall back to the true matrix if anything looks off.
    if tour.size != n:
        return edge_distance.copy()

    # Edges of the current (closed) tour: (i, i+1) and (last, first).
    a = tour[:-1]
    b = tour[1:]
    # closing edge
    a = np.concatenate([a, [tour[-1]]])
    b = np.concatenate([b, [tour[0]]])

    # True lengths of the current tour edges.
    edge_len = edge_distance[a, b]
    mean_len = float(np.mean(edge_len)) if edge_len.size else 1.0
    if mean_len <= 0 or not np.isfinite(mean_len):
        mean_len = 1.0

    # Per-edge usage from the symmetric count matrix (take upper triangle value).
    used = edge_n_used[a, b]
    used = np.maximum(used, 0.0)

    # Base penalty fraction of the mean edge length. We want the penalized tour
    # to be clearly worse than alternatives so 2-opt must leave this basin.
    base_frac = 0.5
    # Extra push for edges that have been reused many times (stagnation escape).
    extra_frac = 0.08 * used

    penalty = (base_frac + extra_frac) * mean_len

    # Build the perturbed matrix: start from true distances.
    out = edge_distance.astype(np.float64, copy=True)
    out[a, b] = edge_distance[a, b] + penalty
    out[b, a] = edge_distance[b, a] + penalty

    # Keep it finite and non-negative.
    out = np.where(np.isfinite(out), out, edge_distance)
    out = np.maximum(out, 0.0)

    # Ensure symmetry (edge_distance is symmetric; our writes are symmetric).
    out = 0.5 * (out + out.T)

    return out
# EVOLVE-BLOCK-END
