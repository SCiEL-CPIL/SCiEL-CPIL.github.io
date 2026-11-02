"""Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
- Decouple the outer hexagon from the search: for a placement of the inner
  hexagons, the minimum enclosing hexagon side is computed analytically by
  sweeping the outer rotation angle. This reduces the optimization dimension
  and lets the optimizer focus purely on packing density.
- Start from a dense hexagonal flower (1 + 6 ring + 5 outer) which is the
  natural dense packing for 12 hexagons, then refine with a time-bounded
  Nelder-Mead and perturb search that minimizes the enclosing side directly.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before the 150s kill

    # Regular hexagon vertices (center (0,0), side 1). Flat-top orientation:
    # the long axis is along x, width=2, height=sqrt(3).
    HEX_VERTS = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2.0],
        [-0.5, np.sqrt(3) / 2.0],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3) / 2.0],
        [0.5, -np.sqrt(3) / 2.0],
    ])
    # Six outward face normals (angle 0,60,...,300), unit length.
    FACE_NORMS = np.array([
        [np.cos(np.radians(60.0 * k)), np.sin(np.radians(60.0 * k))]
        for k in range(6)
    ])

    def rot_mat(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])

    def inner_vertices(x, y, a_deg):
        return (HEX_VERTS * rot_mat(a_deg)).T + np.array([x, y])

    def all_inner_vertices(inner):
        """inner: (N,3) array of [x,y,angle_deg]. Returns (N*6, 2) array."""
        out = np.empty((N * 6, 2))
        for i in range(N):
            x, y, a = inner[i]
            out[6 * i: 6 * i + 6] = inner_vertices(x, y, a)
        return out

    def min_outer_side(verts, angle_deg):
        """Minimum side length of a regular hexagon (center 0, rotation angle_deg)
        that contains all points in verts (an (M,2) array)."""
        R = rot_mat(angle_deg)
        local = verts @ R.T  # rotate points into the hexagon's local frame
        # support in each of the 6 normal directions
        sup = local @ FACE_NORMS.T  # (M, 6)
        maxsup = sup.max(axis=0)    # (6,)
        # apothem >= maxsup  =>  side*sqrt(3)/2 >= maxsup
        return (2.0 / np.sqrt(3.0)) * maxsup.max()

    def enclosing_side(inner):
        """Minimize enclosing side over the outer rotation angle (0..60 deg)."""
        verts = all_inner_vertices(inner)
        best = min_outer_side(verts, 0.0)
        best_a = 0.0
        for a in np.linspace(0.0, 60.0, 25):
            s = min_outer_side(verts, a)
            if s < best - 1e-12:
                best = s
                best_a = a
        # refine around the best angle
        lo, hi = best_a - 2.0, best_a + 2.0
        lo = max(0.0, lo)
        hi = min(60.0, hi)
        for a in np.linspace(lo, hi, 25):
            s = min_outer_side(verts, a)
            if s < best - 1e-12:
                best = s
                best_a = a
        return best

    def make_params(inner):
        return inner.reshape(-1).copy()

    def parse_params(params):
        return np.array(params, dtype=float).reshape(N, 3)

    def objective(params):
        # Fast validity via minimum pairwise vertex distance; fall back to SAT.
        inner = parse_params(params)
        # Vectorized min pairwise distance
        # (N*6, 2)
        allv = all_inner_vertices(inner)
        # Build the 6x6 block of within-hex distances (diagonal) and 30x30 cross blocks.
        min_pair = np.inf
        # within-hexagon min distance (constant for a unit hexagon)
        wv = np.vstack([inner_vertices(0, 0, 0) for _ in range(N)])  # (N*6,2)
        for i in range(N):
            vi = allv[6 * i: 6 * i + 6]
            for j in range(i + 1, N):
                vj = allv[6 * j: 6 * j + 6]
                d = np.linalg.norm(vi[:, None, :] - vj[None, :, :], axis=2)
                m = d.min()
                if m < min_pair:
                    min_pair = m
        # Disjoint iff every pair of hexagons is strictly separated.
        # The min vertex-vertex distance between two convex polygons is 0 iff they
        # touch/cross; but touching can occur edge-edge without a shared vertex, so
        # also check point-in-convex via SAT for safety (cheap for 66 pairs).
        if min_pair > 1e-7:
            # no touching vertices; still guard against edge-edge crossing
            for i in range(N):
                vi = allv[6 * i: 6 * i + 6]
                for j in range(i + 1, N):
                    vj = allv[6 * j: 6 * j + 6]
                    if not disjoint_sat(vi, vj):
                        return enclosing_side(inner) + 5.0
            return enclosing_side(inner)
        # touching/overlapping: penalize
        return enclosing_side(inner) + 5.0

    def disjoint_sat(v1, v2):
        """Separating-axis test for two convex hexagons (6x2 arrays). True if disjoint."""
        # edges -> normals
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
        axes = np.vstack([n1, n2])
        # normalize
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        axes = axes / norms
        p1 = v1 @ axes.T
        p2 = v2 @ axes.T
        lo1, hi1 = p1.min(axis=0), p1.max(axis=0)
        lo2, hi2 = p2.min(axis=0), p2.max(axis=0)
        sep = (hi1 < lo2 - 1e-9) | (hi2 < lo1 - 1e-9)
        return bool(sep.any())

    # ---- Build a dense initial configuration: flower (1 + 6) + 5 outer ----
    def initial_config():
        inner = np.zeros((N, 3))
        inner[0] = [0.0, 0.0, 0.0]
        # ring of 6 at distance 2*sqrt(3) (edges touch)
        d = 2.0 * np.sqrt(3.0)
        for k in range(6):
            a = np.radians(60.0 * k + 30.0)
            inner[k + 1] = [d * np.cos(a), d * np.sin(a), 0.0]
        # outer 5 at distance 4*sqrt(3)
        d2 = 4.0 * np.sqrt(3.0)
        for k in range(5):
            a = np.radians(72.0 * k + 18.0)
            inner[k + 7] = [d2 * np.cos(a), d2 * np.sin(a), 0.0]
        return inner

    best_inner = initial_config()
    best_side = enclosing_side(best_inner)

    params0 = make_params(best_inner)
    cur_params = params0.copy()
    cur_side = best_side

    # ---- Refinement: Nelder-Mead with time budget ----
    rng = np.random.default_rng(12345)

    def time_left():
        return deadline - time.time()

    n_restarts = 4
    for r in range(n_restarts):
        if time_left() < 25.0:
            break
        if r == 0:
            p = cur_params.copy()
        else:
            # perturb best
            p = best_inner.reshape(-1).copy()
            p[:2 * N] += rng.normal(0, 0.3, size=2 * N)   # positions
            p[2 * N:] += rng.normal(0, 8.0, size=N)        # angles
        # keep angles bounded
        p[2 * N:] = ((p[2 * N:] + 180.0) % 360.0) - 180.0

        try:
            res = minimize(objective, p, method='Nelder-Mead',
                           options={'maxiter': 120, 'xatol': 0.02,
                                    'fatol': 0.005, 'adaptive': True,
                                    'maxfev': 400})
        except Exception:
            continue
        inner_c = parse_params(res.x)
        side_c = enclosing_side(inner_c)
        if side_c < cur_side - 1e-9:
            cur_params = res.x.copy()
            cur_side = side_c
        if side_c < best_side - 1e-9:
            best_inner = inner_c.copy()
            best_side = side_c

    # Final polish of the best
    if time_left() > 8.0:
        try:
            res = minimize(objective, make_params(best_inner), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 0.01,
                                    'fatol': 0.002, 'adaptive': True,
                                    'maxfev': 300})
            inner_c = parse_params(res.x)
            side_c = enclosing_side(inner_c)
            if side_c < best_side - 1e-9:
                best_inner = inner_c.copy()
                best_side = side_c
        except Exception:
            pass

    # ---- Compute the exact enclosing side + optimal outer angle ----
    verts = all_inner_vertices(best_inner)
    best_s = float('inf')
    best_a = 0.0
    for a in np.linspace(0.0, 60.0, 121):
        s = min_outer_side(verts, a)
        if s < best_s:
            best_s = s
            best_a = a
    for a in np.linspace(max(0.0, best_a - 1.0), min(60.0, best_a + 1.0), 81):
        s = min_outer_side(verts, a)
        if s < best_s:
            best_s = s
            best_a = a

    # Re-center the inner hexagons so their bounding region is centered at origin.
    # (The enclosing hexagon is centered at (0,0); translating inner points about
    # the origin keeps the containment valid, so we center the vertex cloud.)
    cx = verts[:, 0].mean()
    cy = verts[:, 1].mean()
    # translate each inner hexagon center by (-cx,-cy) to center the cloud
    for i in range(N):
        best_inner[i, 0] -= cx
        best_inner[i, 1] -= cy

    # Recompute enclosing side after re-centering (should be unchanged, but verify)
    verts2 = all_inner_vertices(best_inner)
    final_s = float('inf')
    final_a = 0.0
    for a in np.linspace(0.0, 60.0, 121):
        s = min_outer_side(verts2, a)
        if s < final_s:
            final_s = s
            final_a = a
    for a in np.linspace(max(0.0, final_a - 1.0), min(60.0, final_a + 1.0), 81):
        s = min_outer_side(verts2, a)
        if s < final_s:
            final_s = s
            final_a = a

    inner_out = [(float(p[0]), float(p[1]), float(p[2])) for p in best_inner]
    outer_center = (0.0, 0.0)
    outer_side = float(final_s)
    outer_angle = float(final_a)

    return inner_out, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
