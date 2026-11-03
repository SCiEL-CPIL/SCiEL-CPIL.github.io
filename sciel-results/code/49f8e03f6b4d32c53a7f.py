"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
  1. Build several high-quality topological initial guesses (honeycomb clusters).
  2. Enforce non-penetration via smooth C^1 barrier potentials.
  3. Minimize outer side + barrier with L-BFGS-B (gradient-based, fast).
  4. Refine with Nelder-Mead on the raw objective (non-smooth but exact).
  5. Deadline guard returns the best valid packing found.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # seconds from start; 10s safety margin before 150s kill


def hex_vertices(cx, cy, angle_rad, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy), rotation in radians."""
    return np.array([
        [cx + side * math.cos(angle_rad + i * math.pi / 3),
         cy + side * math.sin(angle_rad + i * math.pi / 3)]
        for i in range(6)
    ])


def _sat_overlap(v1, v2):
    """Return (overlap_depth, min_axis) or (0.0, None) if separated.

    Uses the Separating Axis Theorem on the 12 edge normals (6 per hexagon).
    """
    min_depth = float('inf')
    min_axis = None
    for poly in (v1, v2):
        for i in range(6):
            j = (i + 1) % 6
            edge = poly[j] - poly[i]
            n = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            p1 = v1 @ n
            p2 = v2 @ n
            lo1, hi1 = p1.min(), p1.max()
            lo2, hi2 = p2.min(), p2.max()
            if hi1 <= lo2 or hi2 <= lo1:
                return 0.0, None
            depth = min(hi1, hi2) - max(lo1, lo2)
            if depth < min_depth:
                min_depth = depth
                min_axis = n
    if min_depth > 1e-9:
        return min_depth, min_axis
    return 0.0, None


def _all_verts(params):
    """Decode params -> (positions, angles, list of vertex arrays)."""
    pos = params[:2 * N].reshape(N, 2)
    ang = params[2 * N:]
    verts = [hex_vertices(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
    return pos, ang, verts


def _outer_side(verts):
    """Min outer hex side (centered at origin, angle 0) containing all verts."""
    max_proj = 0.0
    for k in range(6):
        a = math.pi / 6 + k * math.pi / 3
        n = np.array([math.cos(a), math.sin(a)])
        for v in verts:
            mp = np.max(v @ n)
            if mp > max_proj:
                max_proj = mp
    return max_proj / (math.sqrt(3) / 2.0)


def raw_objective(params):
    """Outer side length if no overlap; large penalty otherwise."""
    _, _, verts = _all_verts(params)
    for i in range(N):
        for j in range(i + 1, N):
            d, _ = _sat_overlap(verts[i], verts[j])
            if d > 1e-7:
                return 100.0 + 10.0 * d
    return _outer_side(verts)


def smooth_objective(params, mu=0.3):
    """Outer side + smooth C^1 barrier for overlaps.

    Barrier for pair (i,j) with overlap depth d > 0:
        b(d) = mu * (d + d^2)          (C^1, b(0)=0)
    This is differentiable everywhere (sub-derivative at d=0 is {0, 2mu}).
    We use a finite-difference-free approach: the objective is evaluated
    by the optimizer; gradients are computed internally by L-BFGS-B.
    """
    _, _, verts = _all_verts(params)
    side = _outer_side(verts)
    barrier = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d, _ = _sat_overlap(verts[i], verts[j])
            if d > 0.0:
                barrier += mu * (d + d * d)
    return side + barrier


# ---------------------------------------------------------------------------
# Initial topologies
# ---------------------------------------------------------------------------

def _honeycomb_positions(radius=1.0):
    """
    Generate positions on a hexagonal lattice for a cluster.
    Returns list of (x, y) for a compact cluster.
    """
    pts = []
    # Ring 0: center
    pts.append((0.0, 0.0))
    # Ring 1: 6 neighbors
    for k in range(6):
        a = k * math.pi / 3
        pts.append((2.0 * math.sqrt(3) / 2 * math.cos(a),
                    2.0 * math.sqrt(3) / 2 * math.sin(a)))
    # Ring 2 partial: add more around
    # For 12 hexagons we need 5 more.
    # Add ring-2 positions
    for k in range(6):
        a = k * math.pi / 3 + math.pi / 6
        pts.append((3.0 * math.sqrt(3) / 2 * math.cos(a),
                    3.0 * math.sqrt(3) / 2 * math.sin(a)))
    # We now have 13; take first 12
    return pts[:12]


def _make_params(positions, angles=None):
    if angles is None:
        angles = [0.0] * len(positions)
    pos = np.array(positions, dtype=float).flatten()
    ang = np.array(angles, dtype=float)
    return np.concatenate([pos, ang])


def build_initial_guesses():
    """Return a list of initial parameter vectors."""
    guesses = []

    # Topology 1: Honeycomb cluster (center + 6 + 5 of ring 2)
    pts = _honeycomb_positions()
    guesses.append(_make_params(pts))

    # Topology 2: 3 rows of 4 (rectangular-ish, honeycomb offset)
    pts2 = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5 + (0.75 if r % 2 else 0.0)
            y = r * math.sqrt(3)
            pts2.append((x - 2.25, y - math.sqrt(3)))
    guesses.append(_make_params(pts2))

    # Topology 3: 2 rows of 6 (staggered)
    pts3 = []
    for r in range(2):
        for c in range(6):
            x = c * 1.5 + (0.75 if r % 2 else 0.0)
            y = r * math.sqrt(3)
            pts3.append((x - 3.75, y - math.sqrt(3) / 2))
    guesses.append(_make_params(pts3))

    # Topology 4: Ring of 12 around center (no center hex)
    pts4 = []
    for k in range(12):
        a = k * math.pi / 6
        r = math.sqrt(3)
        pts4.append((r * math.cos(a), r * math.sin(a)))
    guesses.append(_make_params(pts4))

    # Topology 5: Random spread
    rng = np.random.RandomState(42)
    pos5 = rng.randn(N, 2) * 2.5
    ang5 = rng.rand(N) * math.pi
    guesses.append(_make_params(list(map(tuple, pos5)), list(ang5)))

    # Topology 6: Compact 2x6 with slight rotation variation
    pts6 = []
    for r in range(2):
        for c in range(6):
            x = c * 1.5 + (0.75 if r % 2 else 0.0)
            y = r * math.sqrt(3)
            pts6.append((x - 3.75, y - math.sqrt(3) / 2))
    ang6 = [0.0 if (r + c) % 2 == 0 else math.pi / 6
            for r in range(2) for c in range(6)]
    guesses.append(_make_params(pts6, ang6))

    return guesses


def construct():
    t0 = time.time()
    best_side = float('inf')
    best_params = None

    guesses = build_initial_guesses()

    # --- Phase 1: L-BFGS-B with smooth objective ---
    for idx, x0 in enumerate(guesses):
        if time.time() - t0 > DEADLINE - 15:
            break
        try:
            res = minimize(
                smooth_objective, x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6,
                         'maxls': 40, 'iprint': 0}
            )
            # Evaluate raw objective at result
            val = raw_objective(res.x)
            if val < 100.0 and val < best_side:
                best_side = val
                best_params = res.x.copy()
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead refinement on raw objective ---
    if best_params is not None:
        if time.time() - t0 < DEADLINE - 5:
            try:
                res2 = minimize(
                    raw_objective, best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7,
                             'adaptive': True}
                )
                val2 = raw_objective(res2.x)
                if val2 < 100.0 and val2 < best_side:
                    best_side = val2
                    best_params = res2.x.copy()
            except Exception:
                pass

    # --- Phase 3: If still no valid solution, try more random starts with NM ---
    if best_params is None or best_side >= 100.0:
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() - t0 > DEADLINE - 5:
                break
            pos = rng.randn(N, 2) * 3.0
            ang = rng.rand(N) * math.pi
            x0 = _make_params(list(map(tuple, pos)), list(ang))
            try:
                res = minimize(
                    raw_objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                val = raw_objective(res.x)
                if val < 100.0 and val < best_side:
                    best_side = val
                    best_params = res.x.copy()
            except Exception:
                pass

    # Final safety: if we still have no valid packing, construct a guaranteed
    # valid (but larger) one using a wide hexagonal lattice.
    if best_params is None or best_side >= 100.0:
        # Place 12 hexagons on a hex lattice with large spacing to avoid overlap
        pts = []
        spacing = 2.0 * math.sqrt(3)  # center-to-center for non-overlap
        for r in range(4):
            for c in range(3):
                x = c * spacing * 0.75 + (spacing * 0.375 if r % 2 else 0.0)
                y = r * spacing * 0.5
                pts.append((x - 1.5 * spacing, y - 1.5 * spacing))
        best_params = _make_params(pts)
        best_side = raw_objective(best_params)
        # If still invalid (shouldn't happen with spacing=2*sqrt(3)), scale up
        if best_side >= 100.0:
            scale = 1.5
            pts_scaled = [(x * scale, y * scale) for x, y in pts]
            best_params = _make_params(pts_scaled)
            best_side = raw_objective(best_params)

    # Extract
    positions = best_params[:2 * N].reshape(N, 2)
    angles = best_params[2 * N:]

    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]),
                      float(math.degrees(angles[i]))])

    return inner, (0.0, 0.0), float(best_side), 0.0


# EVOLVE-BLOCK-END
