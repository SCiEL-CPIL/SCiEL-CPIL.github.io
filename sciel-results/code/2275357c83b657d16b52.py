"""
Packing 11 unit hexagons in a regular hexagon.

Strategy:
1. Build a few candidate *rigid composites* (fixed relative positions/angles of
   the 11 inner hexagons, e.g. hex-lattice clusters).
2. Global search over the 6 DOF of the composite (x, y, rotation) plus the
   outer hexagon's rotation, using differential evolution.
3. Refine: allow each inner hexagon's angle to vary (semi-rigid), using
   Nelder-Mead from the best DE point.
4. Polish the outer hexagon (center + angle) with a small grid / Nelder-Mead.

The score is the outer side length (lower is better).
"""

import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = math.sqrt(3.0)
DEADLINE = 142.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def hex_verts(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon (CCW)."""
    return [(cx + side * math.cos(angle_rad + i * math.pi / 3.0),
             cy + side * math.sin(angle_rad + i * math.pi / 3.0))
            for i in range(6)]


def _dot(ax, ay, bx, by):
    return ax * bx + ay * by


def required_side(inner, outer_center, outer_angle_deg):
    """
    Minimum side length of an outer regular hexagon (center, angle) that
    contains all inner hexagons (each with side 1 and given angle).
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    ca = math.cos(math.radians(outer_angle_deg))
    sa = math.sin(math.radians(outer_angle_deg))

    # Six outward normals of the outer hexagon (in its local frame)
    normals = [(math.cos(math.radians(30 + k * 60)),
                math.sin(math.radians(30 + k * 60))) for k in range(6)]

    max_proj = 0.0
    for (ix, iy, iang) in inner:
        a = math.radians(iang)
        for k in range(6):
            th = a + k * math.pi / 3.0
            vx = ix + math.cos(th)
            vy = iy + math.sin(th)
            dx = vx - ox
            dy = vy - oy
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            for (nx, ny) in normals:
                p = lx * nx + ly * ny
                if p > max_proj:
                    max_proj = p
    return 2.0 * max_proj / SQRT3


def inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-7):
    ox, oy = outer_center
    ca = math.cos(math.radians(outer_angle_deg))
    sa = math.sin(math.radians(outer_angle_deg))
    apothem = outer_side * SQRT3 / 2.0
    normals = [(math.cos(math.radians(30 + k * 60)),
                math.sin(math.radians(30 + k * 60))) for k in range(6)]
    for (ix, iy, iang) in inner:
        a = math.radians(iang)
        for k in range(6):
            th = a + k * math.pi / 3.0
            vx = ix + math.cos(th)
            vy = iy + math.sin(th)
            dx = vx - ox
            dy = vy - oy
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            for (nx, ny) in normals:
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def _proj_interval(verts, ax, ay):
    vals = [vx * ax + vy * ay for (vx, vy) in verts]
    return min(vals), max(vals)


def sat_overlap(v1, v2, eps=1e-9):
    """True if two convex polygons overlap (touching counts as overlap)."""
    axes = []
    for verts in (v1, v2):
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            axes.append((-ey / L, ex / L))
    for (ax, ay) in axes:
        m1, M1 = _proj_interval(v1, ax, ay)
        m2, M2 = _proj_interval(v2, ax, ay)
        if M1 < m2 + eps or M2 < m1 + eps:
            return False
    return True


def all_disjoint(inner, eps=1e-7):
    n = len(inner)
    verts = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(verts[i], verts[j], eps):
                return False
    return True


# ---------------------------------------------------------------------------
# Composite layouts
# ---------------------------------------------------------------------------

def axial_to_xy(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q * 0.5)
    return x, y


def build_composites():
    """Return list of composites. Each composite is a list of (x, y, angle_deg)
    in the composite's local frame (centered near origin, no global rotation)."""
    composites = []

    # 1) Hexagonal cluster: center + ring of 6 + 4 on top of ring
    ring = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    extra = [(2, 0), (1, 1), (0, 2), (-1, 2)]
    pts = [(0, 0)] + ring + extra
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 2) 3x3 block + 2
    pts = [(q, r) for r in range(3) for q in range(3)] + [(1, 3), (2, 3)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 3) 4 + 3 + 4 rows
    pts = [(q, 0) for q in range(4)] + [(q, 1) for q in range(1, 4)] + \
          [(q, 2) for q in range(4)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 4) 5 + 6 (two long rows)
    pts = [(q, 0) for q in range(5)] + [(q, 1) for q in range(1, 7)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 5) Center + ring + 4 on different side
    extra2 = [(2, 0), (2, -1), (1, -2), (0, -2)]
    pts = [(0, 0)] + ring + extra2
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 6) 2x5 + 1
    pts = [(q, r) for r in range(2) for q in range(5)] + [(2, 2)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 7) 3 + 5 + 3
    pts = [(q, 0) for q in range(1, 4)] + [(q, 1) for q in range(5)] + \
          [(q, 2) for q in range(1, 4)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 8) Star: center + ring + 4 diagonal
    extra3 = [(2, 1), (1, 2), (-1, 2), (-2, 1)]
    pts = [(0, 0)] + ring + extra3
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 9) 4 + 4 + 3
    pts = [(q, 0) for q in range(4)] + [(q, 1) for q in range(4)] + \
          [(q, 2) for q in range(1, 4)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 10) 5 + 4 + 2
    pts = [(q, 0) for q in range(5)] + [(q, 1) for q in range(1, 5)] + \
          [(q, 2) for q in range(2, 4)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 11) 3x4 minus 1 corner + 1
    pts = [(q, r) for r in range(4) for q in range(3) if not (q == 0 and r == 3)]
    pts.append((3, 1))
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 12) Compact: center + ring + 4 forming a 2x2 on one side
    extra4 = [(1, 1), (2, 1), (1, 2), (2, 2)]
    pts = [(0, 0)] + ring + extra4
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 13) 2 + 3 + 3 + 3
    pts = [(q, 0) for q in range(2)] + [(q, 1) for q in range(3)] + \
          [(q, 2) for q in range(3)] + [(q, 3) for q in range(3)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    # 14) 4 + 3 + 4 (symmetric)
    pts = [(q, 0) for q in range(4)] + [(q, 1) for q in range(1, 4)] + \
          [(q, 2) for q in range(4)]
    composites.append([axial_to_xy(q, r) + (0.0,) for q, r in pts])

    return composites


def center_composite(comp):
    """Translate composite so its centroid is at origin."""
    cx = sum(p[0] for p in comp) / len(comp)
    cy = sum(p[1] for p in comp) / len(comp)
    return [(x - cx, y - cy, a) for (x, y, a) in comp]


def rotate_composite(comp, angle_deg):
    """Rotate composite around origin by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for (x, y, ia) in comp:
        out.append((x * ca - y * sa, x * sa + y * ca, ia + angle_deg))
    return out


# ---------------------------------------------------------------------------
# Rigid composite objective (6 DOF: x, y, comp_angle, outer_angle, plus we
# fix outer center at origin WLOG because we can always translate)
# Actually: outer center is free too, but we can set it to the composite's
# centroid after rotation for a tighter bound. Let's keep 4 DOF:
#   comp_x, comp_y, comp_angle_deg, outer_angle_deg
# and place outer center at (comp_x, comp_y) (centroid).
# ---------------------------------------------------------------------------

def rigid_objective(params, comp):
    """params = [cx, cy, comp_angle_deg, outer_angle_deg]"""
    cx, cy, comp_ang, outer_ang = params
    rot = rotate_composite(comp, comp_ang)
    inner = [(x + cx, y + cy, a) for (x, y, a) in rot]
    return required_side(inner, (cx, cy), outer_ang)


def rigid_bounds(comp):
    # Estimate extent
    extent = 0.0
    for (x, y, _) in comp:
        extent = max(extent, math.hypot(x, y) + 2.0)
    return [(-extent, extent), (-extent, extent), (-180, 180), (0, 60)]


# ---------------------------------------------------------------------------
# Semi-rigid refinement: inner angles can vary
# ---------------------------------------------------------------------------

def semi_rigid_objective(params, comp, outer_ang):
    """
    params = [cx, cy, comp_ang, a1, a2, ..., aN]
    comp_ang is the base rotation; ai are per-hexagon angle offsets.
    """
    cx, cy, comp_ang = params[0], params[1], params[2]
    offsets = params[3:]
    rot = rotate_composite(comp, comp_ang)
    inner = [(x + cx, y + cy, a + off) for ((x, y, a), off) in zip(rot, offsets)]
    return required_side(inner, (cx, cy), outer_ang)


# ---------------------------------------------------------------------------
# Polish outer hexagon for a fixed inner layout
# ---------------------------------------------------------------------------

def polish_outer(inner, start_time, deadline):
    """Find best (center, angle) for fixed inner layout."""
    cx0 = sum(x for x, y, _ in inner) / len(inner)
    cy0 = sum(y for x, y, _ in inner) / len(inner)

    best_side = float('inf')
    best_c = (cx0, cy0)
    best_a = 0.0

    # Grid over angles
    for a0 in range(0, 60, 5):
        if time.time() - start_time > deadline - 5:
            break
        # For each angle, optimize center with Nelder-Mead
        def obj_center(p):
            return required_side(inner, (p[0], p[1]), a0)

        x0 = np.array([cx0, cy0])
        try:
            res = minimize(obj_center, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            s = res.fun
            if s < best_side:
                best_side = s
                best_c = (res.x[0], res.x[1])
                best_a = float(a0)
        except Exception:
            pass

    # Fine polish around best
    if time.time() - start_time < deadline - 3:
        def obj_full(p):
            return required_side(inner, (p[0], p[1]), p[2])

        x0 = np.array([best_c[0], best_c[1], best_a])
        try:
            res = minimize(obj_full, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_side:
                best_side = res.fun
                best_c = (res.x[0], res.x[1])
                best_a = float(res.x[2])
        except Exception:
            pass

    return best_c, best_a, best_side


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    composites = [center_composite(c) for c in build_composites()]

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for ci, comp in enumerate(composites):
        if time.time() - start_time > DEADLINE - 20:
            break

        # --- Phase 1: Differential Evolution on rigid 4-DOF ---
        bounds = rigid_bounds(comp)
        best_rigid = None
        best_rigid_val = float('inf')

        if time.time() - start_time < DEADLINE - 15:
            try:
                de_res = differential_evolution(
                    rigid_objective, bounds, args=(comp,),
                    maxiter=80, popsize=15, tol=1e-6,
                    mutation=(0.3, 0.8), recombination=0.7,
                    seed=42 + ci, polish=False,
                    init='sobol'
                )
                if de_res.fun < best_rigid_val:
                    best_rigid_val = de_res.fun
                    best_rigid = de_res.x
            except Exception:
                pass

        # If DE failed, use a coarse grid
        if best_rigid is None:
            best_rigid = np.array([0.0, 0.0, 0.0, 0.0])
            best_rigid_val = rigid_objective(best_rigid, comp)

        # --- Phase 2: Nelder-Mead polish on rigid 4-DOF ---
        if time.time() - start_time < DEADLINE - 10:
            try:
                nm_res = minimize(
                    rigid_objective, best_rigid, args=(comp,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                if nm_res.fun < best_rigid_val:
                    best_rigid_val = nm_res.fun
                    best_rigid = nm_res.x
            except Exception:
                pass

        # Build inner layout from best rigid solution
        cx, cy, comp_ang, outer_ang = best_rigid
        rot = rotate_composite(comp, comp_ang)
        inner = [(x + cx, y + cy, a) for (x, y, a) in rot]

        # --- Phase 3: Semi-rigid refinement (per-hexagon angles) ---
        if time.time() - start_time < DEADLINE - 8:
            x0 = np.concatenate(([cx, cy, comp_ang],
                                 np.zeros(N)))
            try:
                sr_res = minimize(
                    semi_rigid_objective, x0, args=(comp, outer_ang),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                if sr_res.fun < best_rigid_val:
                    best_rigid_val = sr_res.fun
                    cx, cy, comp_ang = sr_res.x[0], sr_res.x[1], sr_res.x[2]
                    offsets = sr_res.x[3:]
                    rot = rotate_composite(comp, comp_ang)
                    inner = [(x + cx, y + cy, a + off)
                             for ((x, y, a), off) in zip(rot, offsets)]
            except Exception:
                pass

        # --- Phase 4: Polish outer hexagon ---
        if time.time() - start_time < DEADLINE - 4:
            c, a, s = polish_outer(inner, start_time, DEADLINE)
            if s < best_rigid_val:
                best_rigid_val = s
                best_center = c
                best_angle = a
            else:
                best_center = (cx, cy)
                best_angle = float(outer_ang)
        else:
            best_center = (cx, cy)
            best_angle = float(outer_ang)
            best_rigid_val = required_side(inner, best_center, best_angle)

        # --- Phase 5: Validate and update global best ---
        if all_disjoint(inner) and inside_outer(inner, best_center, best_rigid_val, best_angle):
            if best_rigid_val < best_side:
                best_side = best_rigid_val
                best_inner = [list(p) for p in inner]
                best_center = best_center
                best_angle = best_angle

    # Fallback: simple hex cluster with no optimization
    if best_inner is None:
        comp = center_composites[0] if 'center_composites' in dir() else \
               center_composite(build_composites()[0])
        inner = [(x, y, a) for (x, y, a) in comp]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = required_side(inner, best_center, best_angle)
        best_inner = [list(p) for p in inner]

    # Final safety check
    if not all_disjoint(best_inner) or not inside_outer(best_inner, best_center, best_side, best_angle):
        # Scale up outer slightly to ensure containment
        best_side *= 1.01
        if not inside_outer(best_inner, best_center, best_side, best_angle):
            # Last resort: use a very large outer hexagon
            best_side = 10.0
            best_center = (0.0, 0.0)
            best_angle = 0.0

    return best_inner, best_center, best_side, best_angle
