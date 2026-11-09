"""QAOA angle design via graph-property-based schedule interpolation."""
import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()
_TIME_MARGIN = 5.0  # safety margin before evaluator kills


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _TIME_MARGIN - _elapsed()


def _graph_properties(n, edges):
    """Compute fast, deterministic graph invariants for schedule selection."""
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Build adjacency matrix (sparse-friendly but n<=10 so dense is fine)
    adj = np.zeros((n, n), dtype=np.int8)
    for (u, v) in edges:
        if u < n and v < n:
            adj[u, v] = 1
            adj[v, u] = 1

    m = int(adj.sum()) // 2
    density = m / (n * (n - 1) / 2.0) if n > 1 else 0.0

    degrees = adj.sum(axis=1).astype(float)
    mean_deg = degrees.mean() if n > 0 else 0.0
    var_deg = degrees.var() if n > 1 else 0.0
    max_deg = float(degrees.max()) if n > 0 else 0.0

    # Triangle density: trace(A^3) / 6
    if n >= 3:
        a2 = adj @ adj
        triangles = int(np.trace(a2 @ adj)) / 6.0
        # Normalize: max possible triangles for n nodes
        max_tri = n * (n - 1) * (n - 2) / 6.0
        tri_density = triangles / max_tri if max_tri > 0 else 0.0
    else:
        tri_density = 0.0

    return density, mean_deg, var_deg, max_deg, tri_density


# Bank of pre-tuned QAOA angle schedules for p=2, indexed by graph "type"
# Each entry: (gamma1, beta1, gamma2, beta2)
# Tuned heuristically based on known QAOA literature for small graphs:
# - Low density / sparse: larger gammas, smaller betas
# - High density / dense: smaller gammas, moderate betas
# - Regular: balanced
# - Irregular (high variance): intermediate
_ANGLE_BANK = {
    # (density_range, var_range): (g1, b1, g2, b2)
    "sparse_low_var":  (0.55, 0.30, 0.35, 0.20),   # sparse, regular
    "sparse_high_var": (0.50, 0.25, 0.30, 0.18),   # sparse, irregular
    "mid_low_var":     (0.45, 0.35, 0.28, 0.22),   # medium density, regular
    "mid_high_var":    (0.40, 0.30, 0.25, 0.20),   # medium density, irregular
    "dense_low_var":   (0.35, 0.40, 0.22, 0.25),   # dense, regular
    "dense_high_var":  (0.30, 0.35, 0.20, 0.22),   # dense, irregular
    "very_dense":      (0.25, 0.45, 0.18, 0.28),   # very dense (near complete)
    "fallback":        (0.40, 0.30, 0.25, 0.20),   # safe default
}


def _select_schedule(density, mean_deg, var_deg, max_deg, tri_density):
    """Select and possibly interpolate between schedules based on graph properties."""
    n = int(max_deg + 0.5)  # rough n estimate not needed; use density directly

    # Classify density
    if density < 0.30:
        density_class = "sparse"
    elif density < 0.60:
        density_class = "mid"
    elif density < 0.85:
        density_class = "dense"
    else:
        density_class = "very_dense"

    # Classify regularity
    # Normalize variance by mean for scale-invariance
    cv = var_deg / (mean_deg + 1e-9)  # coefficient of variation
    if cv < 0.3:
        var_class = "low_var"
    else:
        var_class = "high_var"

    if density_class == "very_dense":
        key = "very_dense"
    else:
        key = f"{density_class}_{var_class}"

    if key not in _ANGLE_BANK:
        key = "fallback"

    g1, b1, g2, b2 = _ANGLE_BANK[key]

    # Fine-tune via triangle density: more triangles -> slightly smaller gammas
    # (triangles create frustration that reduces effective gap)
    tri_shift = -0.05 * tri_density
    g1 += tri_shift
    g2 += tri_shift * 0.7

    # Fine-tune via max degree: high-degree nodes benefit from slightly larger betas
    deg_shift = 0.02 * (max_deg - mean_deg) if mean_deg > 0 else 0.0
    b1 += deg_shift
    b2 += deg_shift * 0.6

    # Clamp to reasonable ranges
    g1 = np.clip(g1, 0.1, 1.0)
    b1 = np.clip(b1, 0.05, 0.8)
    g2 = np.clip(g2, 0.05, 0.8)
    b2 = np.clip(b2, 0.05, 0.8)

    return np.array([[g1, b1], [g2, b2]], dtype=np.float64)


def angles(n, edges, p):
    """Return QAOA angles of shape (p, 2) for the given graph.
    
    Args:
        n: number of qubits / graph vertices
        edges: iterable of (u, v) edge pairs
        p: number of QAOA layers (expected to be 2)
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Deadline guard
    if _time_left() < 0.1:
        return np.tile([0.35, 0.25], (p, 1))

    try:
        # Convert edges to list for consistent processing
        edge_list = list(edges)

        # Compute graph properties (fast for n <= 10)
        density, mean_deg, var_deg, max_deg, tri_density = _graph_properties(n, edge_list)

        # Select schedule from bank
        result = _select_schedule(density, mean_deg, var_deg, max_deg, tri_density)

        # Ensure shape is (p, 2); if p != 2, tile or truncate
        if p == 2:
            return result
        elif p > 2:
            # For extra layers, interpolate toward fixed-point (0, 0)
            rows = [result[0], result[1]]
            for i in range(2, p):
                frac = 1.0 / (i + 1)
                rows.append(result[1] * frac)
            return np.array(rows, dtype=np.float64)
        else:
            # p == 1
            return result[:1]

    except Exception:
        # Fallback: safe default angles
        return np.tile([0.35, 0.25], (p, 1))
