"""
Pack 12 unit regular hexagons inside a minimal regular hexagon.
Radial Slicing: 1 center + 5 inner + 6 outer in concentric rings with
6-fold-symmetric angular offsets. Only 3 continuous parameters are
searched (two ring radii, one global rotation); per-hexagon rotations
are optimized by a fast local refinement loop.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3.0)
TAU = 2.0 * math.pi


def construct():
    t0 = time.time()

    # ---------- Geometry helpers ----------
    def hex_verts(cx, cy, ang_deg):
        a = np.deg2rad(ang_deg)
        base = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((cx + np.cos(base), cy + np.sin(base)))

    def hex_support(v, directions):
        """Max dot product of vertices with a set of unit directions.
        directions: (m, 2) array."""
        return v @ directions.T  # (m,)

    def outer_side_from_points(points, phi_deg):
        """Minimum side length of a regular hexagon (rotation phi_deg)
        that contains all points in `points` (n, 2). Returns (side, cx, cy)."""
        phi = np.deg2rad(phi_deg)
        # 3 unique face-normal directions (the other 3 are negatives)
        angs = phi + np.pi / 6 + np.array([0.0, np.pi / 3, 2.0 * np.pi / 3])
        dirs = np.column_stack((np.cos(angs), np.sin(angs)))
        proj = points @ dirs.T  # (n, 3)
        pmax = proj.max(axis=0)
        pmin = proj.min(axis=0)
        ranges = pmax - pmin
        side = float(ranges.max() / SQRT3)
        mids = (pmax + pmin) / 2.0
        # Solve A @ c = mids where A = [dirs] (3x2)
        c, _, _, _ = np.linalg.lstsq(dirs, mids, rcond=None)
        return side, float(c[0]), float(c[1])

    def min_outer_side(inner):
        """Scan outer rotation phi in [0,60) to minimize side length."""
        all_verts = np.vstack([hex_verts(h[0], h[1], h[2]) for h in inner])
        best = (float('inf'), 0.0, 0.0, 0.0)
        for phi_deg in np.arange(0.0, 60.0, 0.5):
            side, cx, cy = outer_side_from_points(all_verts, phi_deg)
            if side < best[0]:
                best = (side, phi_deg, cx, cy)
        # refine
        if 0.0 < best[1] < 60.0:
            for phi_deg in np.arange(best[1] - 0.5, best[1] + 0.5, 0.05):
                if phi_deg < 0.0 or phi_deg >= 60.0:
                    continue
                side, cx, cy = outer_side_from_points(all_verts, phi_deg)
                if side < best[0]:
                    best = (side, phi_deg, cx, cy)
        return best

    # ---------- Radial slicing configuration ----------
    # 1 center at (0,0)
    # 5 inner ring at radius r1, angles: 60*k + off1 for k=0..4
    # 6 outer ring at radius r2, angles: 60*k + off2 for k=0..5
    def build_inner(r1, r2, off1_deg, off2_deg):
        inner = [[0.0, 0.0, 0.0]]  # center
        for k in range(5):
            a = np.deg2rad(60.0 * k + off1_deg)
            inner.append([r1 * math.cos(a), r1 * math.sin(a), 0.0])
        for k in range(6):
            a = np.deg2rad(60.0 * k + off2_deg)
            inner.append([r2 * math.cos(a), r2 * math.sin(a), 0.0])
        return inner

    # ---------- Objective for the 3-parameter radial search ----------
    def radial_objective(p):
        r1, r2, off1 = p[0], p[1], p[2]
        off2 = off1 + 30.0  # outer ring offset tied to inner for symmetry
        inner = build_inner(r1, r2, off1, off2)

        # Quick overlap check: for hexagons of side 1, center-to-center
        # distance must be at least 2 (touching) when aligned.
        # Use a conservative lower bound: min distance between centers
        # should be >= 2.0 - small tolerance for rotated cases.
        # For speed, use a simple distance check with a permissive bound.
        pts = np.array([[h[0], h[1]] for h in inner])
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d < 1.99:  # conservative: allow slight overlap in objective
                    penalty = (1.99 - d) ** 2
                    return 1e6 + penalty

        side, phi, cx, cy = min_outer_side(inner)
        return side

    # ---------- Fast per-hexagon rotation refinement ----------
    def refine_rotations(inner, max_iter=30):
        """Try small rotation tweaks per hexagon to reduce side length."""
        current = [list(h) for h in inner]
        best_side, _, _, _ = min_outer_side(current)
        for _ in range(max_iter):
            if time.time() - t0 > DEADLINE:
                break
            improved = False
            for i in range(N):
                base_angle = current[i][2]
                for da in [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0]:
                    trial = [list(h) for h in current]
                    trial[i][2] = base_angle + da
                    # quick overlap check
                    pts = np.array([[h[0], h[1]] for h in trial])
                    ok = True
                    for a in range(N):
                        for b in range(a + 1, N):
                            if math.hypot(pts[a, 0] - pts[b, 0],
                                          pts[a, 1] - pts[b, 1]) < 1.99:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    side, _, _, _ = min_outer_side(trial)
                    if side < best_side - 1e-7:
                        best_side = side
                        current = trial
                        improved = True
                        break
            if not improved:
                break
        return current, best_side

    # ---------- Main search ----------
    best_side = float('inf')
    best_inner = None
    best_phi = 0.0
    best_center = (0.0, 0.0)

    def update_best(inner):
        nonlocal best_side, best_inner, best_phi, best_center
        side, phi, cx, cy = min_outer_side(inner)
        if side < best_side:
            best_side = side
            best_inner = [list(h) for h in inner]
            best_phi = phi
            best_center = (cx, cy)

    # --- Phase 1: coarse grid over (r1, r2, off1) ---
    r1_vals = np.arange(2.0, 4.5, 0.25)
    r2_vals = np.arange(3.5, 6.5, 0.25)
    off1_vals = np.arange(0.0, 60.0, 10.0)

    for r1 in r1_vals:
        if time.time() - t0 > DEADLINE - 30.0:
            break
        for r2 in r2_vals:
            for off1 in off1_vals:
                if time.time() - t0 > DEADLINE - 30.0:
                    break
                inner = build_inner(r1, r2, off1, off1 + 30.0)
                update_best(inner)

    # --- Phase 2: local optimization from best grid point ---
    if best_inner is not None:
        # Extract approximate (r1, r2, off1) from best_inner
        # Inner ring: hexagons 1-5
        inner_ring_pts = np.array([[best_inner[i][0], best_inner[i][1]] for i in range(1, 6)])
        r1_est = float(np.mean(np.linalg.norm(inner_ring_pts, axis=1)))
        outer_ring_pts = np.array([[best_inner[i][0], best_inner[i][1]] for i in range(6, 12)])
        r2_est = float(np.mean(np.linalg.norm(outer_ring_pts, axis=1)))
        off1_est = float(np.degrees(np.arctan2(inner_ring_pts[0, 1], inner_ring_pts[0, 0]))) % 60.0

        def opt_obj(p):
            r1, r2, off1 = p
            if r1 < 1.5 or r1 > 6.0 or r2 < 2.5 or r2 > 8.0:
                return 1e6
            return radial_objective([r1, r2, off1])

        x0 = np.array([r1_est, r2_est, off1_est])
        try:
            res = minimize(opt_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side:
                r1, r2, off1 = res.x
                inner = build_inner(r1, r2, off1, off1 + 30.0)
                update_best(inner)
        except Exception:
            pass

    # --- Phase 3: per-hexagon rotation refinement ---
    if best_inner is not None:
        refined, refined_side = refine_rotations(best_inner, max_iter=20)
        if refined_side < best_side:
            best_inner = refined
            best_side = refined_side
            best_side, best_phi, best_center[0], best_center[1] = min_outer_side(best_inner)

    # --- Fallback if nothing valid ---
    if best_inner is None:
        best_inner = build_inner(2.0, 4.0, 0.0, 30.0)
        best_side, best_phi, cx, cy = min_outer_side(best_inner)
        best_center = (cx, cy)

    # Final recompute
    final_side, final_phi, final_cx, final_cy = min_outer_side(best_inner)
    return best_inner, (float(final_cx), float(final_cy)), float(final_side), float(final_phi)


# EVOLVE-BLOCK-END
