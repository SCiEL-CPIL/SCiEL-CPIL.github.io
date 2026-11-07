"""Guided local search penalty rule for the TSP (combined approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Hard wall-clock deadline (well inside the 330 s evaluator kill limit)
_DEADLINE_S = 300.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.monotonic()
    return time.monotonic() - _START


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we ever approach the limit, return the safe base matrix.
    if _elapsed() > _DEADLINE_S - 30.0:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()

    n = D.shape[0]

    # Guard against non-finite input; fall back to the (clean) base matrix.
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base

    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)

    # Reference scale: median of edge lengths in the current local-optimal tour.
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.ndim == 1 and tour.size >= 2:
        m = tour.size
        idx_i = tour
        idx_j = np.roll(tour, -1)
        tour_edges = D[idx_i, idx_j]
        finite_tour = tour_edges[np.isfinite(tour_edges)]
        if finite_tour.size > 0:
            ref = float(np.median(finite_tour))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Determine the current round index from the maximum usage count.
    # This gives us a proxy for how many rounds have passed.
    max_used = float(used.max()) if used.size > 0 else 0.0
    round_idx = int(max_used) if max_used > 0 else 0

    # Adaptive kick magnitude: stronger early, weaker later.
    # Use a smooth decay based on round_idx to avoid discontinuities.
    # kick = base_kick * exp(-decay_rate * round_idx)
    base_kick = 0.4
    decay_rate = 0.15
    kick = base_kick * np.exp(-decay_rate * round_idx)

    # Smooth repulsive force field: exponential saturation penalty
    # This creates a continuous "exclusion zone" where the penalty decays
    # smoothly with distance from the tour in the metric space.
    # penalty = max_p * (1 - exp(-k * used / ref))
    # The exponential form ensures:
    #   - Small usage -> small penalty (linear regime)
    #   - Large usage -> penalty saturates at max_p (bounded)
    #   - Smooth transition (no discontinuities)
    k = 1.2
    max_penalty = 0.7
    scaled = used / ref
    penalty = max_penalty * (1.0 - np.exp(-k * scaled))
    penalty = np.maximum(penalty, 0.0)

    # Tour-aware boost: extra penalty for edges in the current local-opt tour
    # This directly targets the local optimum to help escape it
    tour_boost = 0.3
    if tour.ndim == 1 and tour.size >= 2:
        for i in range(tour.size - 1):
            a, b = tour[i], tour[i + 1]
            if 0 <= a < n and 0 <= b < n:
                penalty[a, b] += tour_boost
                penalty[b, a] += tour_boost
        # Close the tour
        a, b = tour[-1], tour[0]
        if 0 <= a < n and 0 <= b < n:
            penalty[a, b] += tour_boost
            penalty[b, a] += tour_boost

    # Hard cap to prevent extreme distortions
    penalty = np.minimum(penalty, 1.5)

    # Adaptive kick: add extra penalty to tour edges that scales with round
    # This helps escape local optima more aggressively in early rounds
    if tour.ndim == 1 and tour.size >= 2:
        for i in range(tour.size - 1):
            a, b = tour[i], tour[i + 1]
            if 0 <= a < n and 0 <= b < n:
                penalty[a, b] += kick
                penalty[b, a] += kick
        a, b = tour[-1], tour[0]
        if 0 <= a < n and 0 <= b < n:
            penalty[a, b] += kick
            penalty[b, a] += kick

    out = D * (1.0 + penalty)

    # Enforce exact symmetry (defensive; both inputs are symmetric).
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
