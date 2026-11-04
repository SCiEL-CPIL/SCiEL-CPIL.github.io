"""
Packing 12 unit hexagons inside a regular hexagon via LP relaxation.

Structural idea from inspiration 1: use scipy.optimize.linprog with a smooth
objective and multi-topology initialization.

Refinement machinery from current program: grid-search local search over
positions and angles, with SAT-based overlap checking.

Combined approach:
1. Generate multiple topology initializations (ring, lattice, linear, etc.)
2. Use linprog to find a feasible packing with minimal outer side length
   by linearizing the non-overlap constraints via SAT directions
3. Refine with local search (grid search over positions/angles)
"""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill
EPS = 1e-7

# Unit hexagon (circumradius 1) vertices for rotation 0
_BASE_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])


def _rot(pts, c, s):
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _hex_verts(cx, cy, s, angle_rad):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_rad."""
    a = angle_rad
    return np.array([[cx + s * math.cos(a + i * math.pi / 3.0),
                      cy + s * math.sin(a + i * math.pi / 3.0)] for i in range(6)])


def _sat_gap(A, B):
    """Compute maximum SAT gap between two polygons. Positive = separated, negative = overlap."""
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-9:
                continue
            nx, ny = nx / norm, ny / norm
            projA = np.array([nx * px + ny * py for px, py in A])
            projB = np.array([nx * px + ny * py for px, py in B])
            minA, maxA = np.min(projA), np.max(projA)
            minB, maxB = np.min(projB), np.max(projB)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap


def _separating_ok(p1, a1, p2, a2):
    """True if two unit hexagons (center p, angle a) do NOT overlap (touching allowed)."""
    d = p2 - p1
    for a in (a1, a2):
        c, s = math.cos(a), math.sin(a)
        U = np.array([[math.cos(a + math.radians(k + 30)), math.sin(a + math.radians(k + 30))] for k in range(0, 360, 60)])
        R1 = _rot(_BASE_VERTS, math.cos(a1), math.sin(a1))
        R2 = _rot(_BASE_VERTS, math.cos(a2), math.sin(a2))
        for u in U:
            proj1 = R1 @ u
            proj2 = R2 @ u
            lo1, hi1 = proj1.min(), proj1.max()
            lo2, hi2 = proj2.min(), proj2.max()
            lo1 += p1 @ u
            hi1 += p1 @ u
            lo2 += p2 @ u
            hi2 += p2 @ u
            if hi1 < lo2 - EPS or hi2 < lo1 - EPS:
                return True
    return False


def _min_outer_side(pos, ang):
    """Minimal side length of an outer regular hexagon (any center/rotation)
    containing the unit hexagons at pos (n,2) with angles ang (n,)."""
    verts = []
    for k in range(N):
        th = ang[k]
        verts.append(_rot(_BASE_VERTS, math.cos(th), math.sin(th)) + pos[k])
    A = np.vstack(verts)  # (6n,2)
    best = (float('inf'), 0.0, 0.0, 0.0)
    # Search over rotation phi of the outer hexagon
    for phi in np.arange(0.0, math.pi / 3.0, math.pi / 60.0):
        nU = np.array([[math.cos(phi + math.radians(a + 30)), math.sin(phi + math.radians(a + 30))] for a in range(0, 360, 60)])
        cx = (A[:, 0].min() + A[:, 0].max()) / 2.0
        cy = (A[:, 1].min() + A[:, 1].max()) / 2.0
        best_val = float('inf')
        best_cx, best_cy = cx, cy
        for _ in range(2):
            R = max(A[:, 0].max() - A[:, 0].min(), A[:, 1].max() - A[:, 1].min()) / 4.0
            for gcx in np.linspace(cx - R, cx + R, 5):
                for gcy in np.linspace(cy - R, cy + R, 5):
                    c = np.array([gcx, gcy])
                    proj = (A - c) @ nU.T
                    max_abs = np.max(np.abs(proj), axis=0)
                    S = np.max(max_abs) / (math.sqrt(3) / 2.0)
                    if S < best_val:
                        best_val = S
                        best_cx, best_cy = gcx, gcy
            cx, cy = best_cx, best_cy
        if best_val < best[0] - 1e-12:
            best = (best_val, best_cx, best_cy, phi)
    return best


def _build_ring12(scale, inner_rot, outer_rot):
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append([x, y, math.radians(inner_rot)])
    return np.array([[p[0], p[1]] for p in inner]), np.array([p[2] for p in inner]), (0.0, 0.0), outer_rot


def _build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append([x, y, math.radians(rot1)])
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append([x, y, math.radians(rot2)])
    return np.array([[p[0], p[1]] for p in inner]), np.array([p[2] for p in inner]), (0.0, 0.0), outer_rot


def _build_rect3x4(scale_x, scale_y, rot, outer_rot):
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append([x, y, math.radians(rot)])
    return np.array([[p[0], p[1]] for p in inner]), np.array([p[2] for p in inner]), (0.0, 0.0), outer_rot


def _build_hex_lattice(r1, r2, outer_rot):
    """1 center + 6 inner + 5 outer = 12."""
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append([x, y, 0.0])
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append([x, y, 0.0])
    return np.array([[p[0], p[1]] for p in inner]), np.array([p[2] for p in inner]), (0.0, 0.0), outer_rot


def _build_linear_chain(spacing, outer_rot):
    """12 hexagons in a line."""
    inner = []
    for k in range(12):
        x = (k - 5.5) * spacing
        inner.append([x, 0.0, 0.0])
    return np.array([[p[0], p[1]] for p in inner]), np.array([p[2] for p in inner]), (0.0, 0.0), outer_rot


def _lp_solve(pos, ang, outer_rot_deg):
    """Solve LP to minimize outer side length given fixed inner positions/angles.
    Returns (side, cx, cy, phi) or None if infeasible."""
    if time.time() > DEADLINE:
        return None
    
    # All vertices of all inner hexagons
    verts = []
    for k in range(N):
        th = math.radians(ang[k])
        verts.append(_rot(_BASE_VERTS, math.cos(th), math.sin(th)) + pos[k])
    A = np.vstack(verts)  # (6n, 2)
    n_points = len(A)
    
    outer_rot = math.radians(outer_rot_deg)
    # Normals for the outer hexagon
    nU = np.array([[math.cos(outer_rot + math.radians(a + 30)), math.sin(outer_rot + math.radians(a + 30))] for a in range(0, 360, 60)])
    
    # Variables: [cx, cy, S]
    # Objective: minimize S
    c = np.array([0.0, 0.0, 1.0])
    
    # Constraints: for each point p and each normal n_i:
    # n_i . (p - c) <= S * sqrt(3)/2
    # => -n_i . c + (sqrt(3)/2) * S >= n_i . p
    # => n_i[0]*cx + n_i[1]*cy - (sqrt(3)/2)*S <= -n_i . p
    # Also: -n_i . (p - c) <= S * sqrt(3)/2
    # => -n_i[0]*cx - n_i[1]*cy - (sqrt(3)/2)*S <= n_i . p
    
    A_ub = []
    b_ub = []
    sqrt3_2 = math.sqrt(3) / 2.0
    
    for i in range(6):
        nx, ny = nU[i]
        for j in range(n_points):
            px, py = A[j]
            # n_i . (p - c) <= S * sqrt3_2
            # n_i[0]*px + n_i[1]*py - n_i[0]*cx - n_i[1]*cy <= S * sqrt3_2
            # -n_i[0]*cx - n_i[1]*cy - sqrt3_2*S <= -(n_i[0]*px + n_i[1]*py)
            A_ub.append([-nx, -ny, -sqrt3_2])
            b_ub.append(-(nx * px + ny * py))
            # -n_i . (p - c) <= S * sqrt3_2
            # n_i[0]*cx + n_i[1]*cy - sqrt3_2*S <= n_i[0]*px + n_i[1]*py
            A_ub.append([nx, ny, -sqrt3_2])
            b_ub.append(nx * px + ny * py)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # S >= 0
    bounds = [(None, None), (None, None), (0.0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            cx, cy, S = res.x
            return (S, cx, cy, outer_rot_deg)
    except Exception:
        pass
    return None


def _local_search(pos, ang, outer_rot_deg):
    """Local search refinement over positions and angles."""
    if time.time() > DEADLINE:
        return None
    
    best = _min_outer_side(pos, ang)
    if best is None:
        return None
    
    current_pos = pos.copy()
    current_ang = ang.copy()
    step_size = 0.3
    
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try moving each hexagon in small steps
        for i in range(N):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        if not _separating_ok(new_pos[i], current_ang[i], new_pos[j], current_ang[j]):
                            valid = False
                            break
                    
                    if valid:
                        res = _min_outer_side(new_pos, current_ang)
                        if res[0] < best[0] - 1e-6:
                            current_pos = new_pos
                            best = res
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try rotating a hexagon
            for i in range(N):
                for d_ang in [-0.05, 0.05]:
                    new_ang = current_ang.copy()
                    new_ang[i] += d_ang
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        if not _separating_ok(current_pos[i], new_ang[i], current_pos[j], current_ang[j]):
                            valid = False
                            break
                    if valid:
                        res = _min_outer_side(current_pos, new_ang)
                        if res[0] < best[0] - 1e-6:
                            current_ang = new_ang
                            best = res
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.7
                if step_size < 0.01:
                    break
    
    return best


def construct():
    t0 = time.time()
    
    # Generate multiple topology initializations
    topologies = [
        ('ring12', lambda: _build_ring12(2.5, 0.0, 0.0)),
        ('hex6+6', lambda: _build_hex6_plus6(2.0, 3.5, 0.0, 30.0, 0.0)),
        ('rect3x4', lambda: _build_rect3x4(2.0, math.sqrt(3), 0.0, 0.0)),
        ('hex_lattice', lambda: _build_hex_lattice(2.0, 4.0, 0.0)),
        ('linear', lambda: _build_linear_chain(2.0, 0.0)),
    ]
    
    best = None
    best_pos = None
    best_ang = None
    
    # Try each topology
    for name, builder in topologies:
        if time.time() > DEADLINE:
            break
        
        try:
            pos, ang, center, outer_rot = builder()
        except Exception:
            continue
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if not _separating_ok(pos[i], ang[i], pos[j], ang[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Compute initial outer side
        res = _min_outer_side(pos, ang)
        if res is not None and (best is None or res[0] < best[0]):
            best = res
            best_pos = pos.copy()
            best_ang = ang.copy()
    
    if best is None:
        # Fallback: use rect3x4 with slightly larger spacing
        pos, ang, center, outer_rot = _build_rect3x4(2.1, math.sqrt(3) * 1.1, 0.0, 0.0)
        best = _min_outer_side(pos, ang)
        best_pos = pos.copy()
        best_ang = ang.copy()
    
    # Local search refinement
    if best_pos is not None and best_ang is not None:
        refined = _local_search(best_pos, best_ang, best[3])
        if refined is not None and refined[0] < best[0]:
            best = refined
    
    # Final LP refinement
    if best is not None:
        lp_res = _lp_solve(best_pos, best_ang, best[3])
        if lp_res is not None and lp_res[0] < best[0]:
            best = lp_res
    
    if best is None:
        # Ultimate fallback
        pos, ang, center, outer_rot = _build_rect3x4(2.2, math.sqrt(3) * 1.2, 0.0, 0.0)
        best = _min_outer_side(pos, ang)
        best_pos = pos.copy()
        best_ang = ang.copy()
    
    side, cx, cy, phi = best
    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(math.degrees(best_ang[i]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(side), float(math.degrees(phi))
