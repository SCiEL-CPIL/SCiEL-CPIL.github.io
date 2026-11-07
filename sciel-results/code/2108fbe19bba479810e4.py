"""Guided local search penalty rule for the TSP (hybrid targeted + historical)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

    # Build a mask of edges present in the current local-optimum tour.
    tour_mask = np.zeros_like(D, dtype=bool)
    t = local_opt_tour
    if t is not None:
        try:
            t_arr = np.asarray(t, dtype=int)
            if t_arr.ndim == 1 and len(t_arr) >= 3:
                for i in range(len(t_arr)):
                    a = int(t_arr[i])
                    b = int(t_arr[(i + 1) % len(t_arr)])
                    if 0 <= a < n and 0 <= b < n and a != b:
                        tour_mask[a, b] = True
                        tour_mask[b, a] = True
        except (ValueError, TypeError, IndexError):
            pass

    # Reference scales: use median of tour edge distances for tour penalty,
    # and median of all used edge distances for historical penalty.
    if tour_mask.any():
        ref_tour = float(np.median(D[tour_mask]))
    else:
        ref_tour = float(np.median(D))
    if not np.isfinite(ref_tour) or ref_tour <= 0.0:
        ref_tour = 1.0

    nz = used > 0.0
    if nz.any():
        ref_hist = float(np.median(D[nz]))
    else:
        ref_hist = float(np.median(D))
    if not np.isfinite(ref_hist) or ref_hist <= 0.0:
        ref_hist = 1.0

    # Penalty parameters:
    # Tour edges: strong, fast-saturating penalty to force 2-opt to restructure.
    k_t = 3.0
    A_t = 2.0
    # Historical (non-tour) used edges: milder, slower-saturating to discourage
    # re-entry into recently-used substructures without overwhelming the search.
    k_h = 1.0
    A_h = 0.5

    penalty = np.zeros_like(D)

    # Historical penalty on non-tour used edges.
    hist_mask = nz & (~tour_mask)
    if hist_mask.any():
        scaled_h = used[hist_mask] / ref_hist
        p_h = A_h * (1.0 - np.exp(-k_h * scaled_h))
        p_h = np.clip(p_h, 0.0, A_h)
        penalty[hist_mask] = p_h

    # Tour penalty: stronger, overwrites any historical penalty on tour edges.
    if tour_mask.any():
        scaled_t = used[tour_mask] / ref_tour
        p_t = A_t * (1.0 - np.exp(-k_t * scaled_t))
        p_t = np.clip(p_t, 0.0, A_t)
        penalty[tour_mask] = p_t

    # Multiply the true distances by (1 + penalty) for penalized edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry.
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
