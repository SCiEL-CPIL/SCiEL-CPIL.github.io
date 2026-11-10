# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 5.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(ang), cy + s * np.sin(ang)])


def _edge_normals(verts):
    e = np.roll(verts, -1, axis=0) - verts
    n = np.column_stack([e[:, 1], -e[:, 0]])
    nrm = np.linalg.norm(n, axis=1)
    nrm[nrm == 0] = 1.0
    return n / nrm[:, None]


def _separation_gap(A, B, nA, nB):
    # A, B: (6,2) vertex arrays; nA, nB: (6,2) normals
    d = B - A[None, :]  # (6,6,2)
    projA = A @ nA.T    # (6,6)
    projB = B @ nB.T
    gap = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            gap[i, j] = np.max(np.abs(projA[i, j] - projB[i, j]))
    # gap[i,j] = max over the 6x6 pair of |proj on nA[i] - proj on nB[j]|
    # separation holds if max over all (i,j) of that gap > 0
    return gap.max()


def _build_inner(theta):
    # theta: (11, 3) = (x, y, angle_deg)
    xs = theta[:, 0]
    ys = theta[:, 1]
    angs = theta[:, 2]
    verts = np.zeros((11, 6, 2))
    norms = np.zeros((11, 6, 2))
    for k in range(11):
        v = _hex_vertices(xs[k], ys[k], 1.0, angs[k])
        verts[k] = v
        norms[k] = _edge_normals(v)
    return verts, norms


def _minimize_side(verts, norms, outer_center, outer_angle):
    # outer hexagon: center (cx,cy), side s, angle a
    # vertices of outer: (cx + s cos(a + 60 i), cy + s sin(...))
    # For a point p to be inside outer hexagon (regular, convex), it must be
    # on the inner side of every edge. Edge from v_i to v_{i+1}; inward normal.
    # We minimize s such that all inner vertices are inside.
    # For fixed angle and center, the minimal s containing all inner vertices:
    #   For each outer edge direction, the supporting function.
    # A regular hexagon centered at C with "radius" (apothem) r = s*sqrt(3)/2.
    # Inward half-space for edge k: n_k . (p - C) <= r, where n_k is inward unit normal.
    # Inward normals point toward center: for edge between v_k, v_{k+1},
    # the outward normal is perpendicular to edge; inward = -outward.
    # The maximum over all inner points and all 6 edges of n_k . (p - C) must be <= r.
    # So minimal r = max_k max_p [ n_k . (p - C) ]. Then s = 2 r / sqrt(3).
    a = np.deg2rad(outer_angle)
    # outward normals of the 6 edges of a unit (s=1) hexagon centered at origin
    # vertices at angle a + 60 i. Edge i from v_i to v_{i+1}.
    # outward normal of edge i points away from center.
    # For a regular hexagon, outward normal of edge i is at angle (a + 60 i + 30)? 
    # Let's compute directly.
    ang = a + np.deg2rad(60.0 * np.arange(6))
    v = np.column_stack([np.cos(ang), np.sin(ang)])  # unit hex vertices
    e = np.roll(v, -1, axis=0) - v
    out_n = np.column_stack([e[:, 1], -e[:, 0]])
    on = np.linalg.norm(out_n, axis=1)
    out_n = out_n / on[:, None]
    # ensure outward (dot with vertex midpoint positive)
    mid = (v + np.roll(v, -1, axis=0)) / 2.0
    sign = np.sign(np.sum(out_n * mid, axis=1))
    out_n = out_n * sign[:, None]
    in_n = -out_n  # inward normals
    # all inner vertices (66 points)
    all_p = verts.reshape(-1, 2)
    # r = max over edges k and points p of in_n[k] . (p - C)
    # (p - C) shape (66,2); in_n (6,2)
    d = all_p - outer_center  # (66,2)
    proj = d @ in_n.T  # (66,6)
    r = proj.max()
    r = max(r, 0.0)
    s = 2.0 * r / np.sqrt(3.0)
    return s


def _separation_violation(verts, norms):
    # return max(0, 1e-6 - gap) summed over pairs (penalty)
    total = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            g = _separation_gap(verts[i], verts[j], norms[i], norms[j])
            if g < 1e-6:
                total += (1e-6 - g) ** 2
    return total


def _objective(params):
    # params: (11*3 + 3) = 36: 11*(x,y,a) + (cx, cy, outer_angle)
    theta = params[:33].reshape(11, 3)
    outer_center = params[33:35]
    outer_angle = params[35]
    verts, norms = _build_inner(theta)
    s = _minimize_side(verts, norms, outer_center, outer_angle)
    sep_viol = _separation_violation(verts, norms)
    # keep inner hexagons near origin to avoid drift
    drift = np.sum(theta[:, 0] ** 2) + np.sum(theta[:, 1] ** 2)
    return s + 1e4 * sep_viol + 1e-3 * drift


def hexagon_packing_11():
    # Honeycomb initial layout: center + 6 inner ring + 4 outer ring
    # For unit hexagons, horizontal spacing 2.5? Actually for flat-top hexagons
    # touching, center distance = 2 * apothem? No: two unit hexagons share an edge
    # when center distance = 2 * (side) * cos(30)?? Let's just use a known honeycomb.
    # Use pointy-top orientation (angle 0 => vertex at angle 0 = right).
    # For a honeycomb of unit hexagons, the lattice vectors are:
    #   a1 = (3/2, sqrt(3)/2) * 2? Let's just place them and let optimizer fix.
    # Simpler: use the classic 7-hex cluster (center + 6) plus 4 more.
    # Center-to-center distance for touching unit hexagons (same orientation) = 2*sqrt(3)/2 *? 
    # Actually the distance between centers of two edge-adjacent unit regular hexagons
    # (flat sides touching) is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732 if
    # they share a full edge. But that's when oriented the same and sharing an edge.
    # Let's place them on a triangular lattice with spacing = sqrt(3) * something.
    # I'll use a generous spacing and let the optimizer tighten.
    h = np.sqrt(3.0) / 2.0  # apothem of unit hexagon
    # Triangular lattice: neighbors at distance 2*h = sqrt(3)
    d = 2.0 * h  # = sqrt(3)
    positions = [
        (0.0, 0.0),
        (d, 0.0),
        (-d, 0.0),
        (0.5 * d, np.sqrt(3) / 2.0 * d),
        (0.5 * d, -np.sqrt(3) / 2.0 * d),
        (-0.5 * d, np.sqrt(3) / 2.0 * d),
        (-0.5 * d, -np.sqrt(3) / 2.0 * d),
        (1.5 * d, np.sqrt(3) / 2.0 * d),
        (1.5 * d, -np.sqrt(3) / 2.0 * d),
        (-1.5 * d, np.sqrt(3) / 2.0 * d),
        (-1.5 * d, -np.sqrt(3) / 2.0 * d),
    ]
    theta0 = np.array([[x, y, 0.0] for (x, y) in positions])
    outer0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([theta0.ravel(), outer0])

    best = None
    best_val = np.inf

    def _eval(x):
        nonlocal best, best_val
        if time.time() > _DEADLINE - _MARGIN:
            return
        try:
            theta = x[:33].reshape(11, 3)
            oc = x[33:35]
            oa = x[35]
            verts, norms = _build_inner(theta)
            s = _minimize_side(verts, norms, oc, oa)
            sep = _separation_violation(verts, norms)
            if sep < 1e-8:
                val = s
                if val < best_val:
                    best_val = val
                    best = (theta.copy(), oc.copy(), oa, s)
        except Exception:
            pass

    # Run a few optimization attempts with different outer angles
    attempts = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 30.0),
        (0.0, 0.0, 15.0),
        (0.0, 0.0, 45.0),
    ]
    for (cx, cy, oa) in attempts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        x0 = np.concatenate([theta0.ravel(), [cx, cy, oa]])
        try:
            res = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 300, "maxfun": 2000, "ftol": 1e-12, "gtol": 1e-8},
                callback=_eval,
            )
            _eval(res.x)
        except Exception:
            pass

    # If no valid best found, fall back to a guaranteed-valid loose packing
    if best is None:
        # Use the seed-style loose packing but with a smaller outer hexagon
        # Ensure validity by making outer hexagon large enough.
        inner = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [-1.25, 2.17, 0.0],
                [1.25, 2.17, 0.0],
                [-1.25, -2.17, 0.0],
                [1.25, -2.17, 0.0],
                [-3.75, 2.17, 0.0],
                [3.75, 2.17, 0.0],
                [-3.75, -2.17, 0.0],
                [3.75, -2.17, 0.0],
            ]
        )
        outer = np.array([0.0, 0.0, 0.0])
        side = 8.0
        return inner, outer, side

    theta, oc, oa, s = best
    inner = theta
    outer = np.array([oc[0], oc[1], oa])
    return inner, outer, float(s)


# EVOLVE-BLOCK-END
