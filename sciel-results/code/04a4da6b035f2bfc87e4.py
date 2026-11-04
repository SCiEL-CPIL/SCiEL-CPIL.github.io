"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 118.0  # Safety margin below 120s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _min_distance_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Compute minimum distance between two hexagons. Negative if overlapping."""
    # Use SAT to find overlap depth or distance
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    
    axes = _axes_from_polygon(verts1) + _axes_from_polygon(verts2)
    max_overlap = -1e10  # Most negative means deepest overlap
    min_gap = 1e10
    
    for axis in axes:
        minA, maxA = _project_polygon(verts1, axis)
        minB, maxB = _project_polygon(verts2, axis)
        
        if maxA <= minB:
            gap = minB - maxA
            if gap < min_gap:
                min_gap = gap
        elif maxB <= minA:
            gap = minA - maxB
            if gap < min_gap:
                min_gap = gap
        else:
            # Overlap on this axis
            overlap = min(maxA, maxB) - max(minA, minB)
            if -overlap > max_overlap:
                max_overlap = -overlap
                
    if max_overlap > 0:
        return max_overlap
    else:
        return min_gap

def _smooth_objective(v):
    """
    Smooth objective for optimization.
    v = [x1, y1, a1, ..., x12, y12, a12, cx, cy, cangle]
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    s = _outer_hex_side(cfg, center, angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_distance_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2], 
                                      cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 0.05: # Only penalize if close or overlapping
                # Quadratic penalty for overlap
                penalty += (0.05 - d) ** 2
                
    return s + 50.0 * penalty

def _exact_objective(v):
    """
    Exact objective for final polish.
    Returns a large number if invalid.
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    if not _is_valid(cfg):
        return 1e6
    
    return _outer_hex_side(cfg, center, angle)

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_symmetric_cluster():
        """
        Create a symmetric cluster of 12 hexagons.
        1 center, 6 inner ring, 5 outer ring.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        
        # Inner ring (6 hexagons)
        # Distance to touch: 2 * apothem = sqrt(3)
        d_inner = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = d_inner * math.cos(theta)
            y = d_inner * math.sin(theta)
            # Rotate to face outward
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
            
        # Outer ring (5 hexagons)
        # Placed in the gaps of the inner ring, further out
        # Distance from center to outer hex center
        # Geometry: Inner hex center at (sqrt(3), 0). Outer hex in gap at angle 30 deg.
        # Let's estimate distance. 
        # The gap between two inner hexagons is at angle 30, 90, etc.
        # Distance to outer hex center should be such that it touches the two adjacent inner hexagons.
        # Let's use a slightly larger radius and let the optimizer adjust.
        d_outer = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = d_outer * math.cos(theta)
            y = d_outer * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
            
        return pts

    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_ring():
        """12 hexagons in a ring."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    # Initial configurations
    inits = [
        make_symmetric_cluster(),
        make_hex_lattice(),
        make_ring()
    ]
    
    # Add some random perturbations to the best symmetric one
    base_cluster = make_symmetric_cluster()
    for _ in range(5):
        perturbed = []
        for p in base_cluster:
            perturbed.append([
                p[0] + random.gauss(0, 0.1),
                p[1] + random.gauss(0, 0.1),
                p[2] + random.gauss(0, 5.0)
            ])
        inits.append(perturbed)

    best_v = None
    best_score = float('inf')
    
    # Global optimization using Differential Evolution on the smooth objective
    # This is robust but slow. We use it to find a good basin.
    # Bounds: x,y in [-10, 10], angle in [-180, 180]
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
    bounds += [(-5, 5), (-5, 5), (-180, 180)]
    
    # We will run a few local optimizations from the initial points
    # instead of a full DE which might be too slow.
    
    for init in inits:
        if time.time() - t0 > TIME_LIMIT - 10.0:
            break
            
        v0 = []
        for p in init:
            v0.extend([p[0], p[1], p[2]])
        v0.extend([0.0, 0.0, 0.0])
        
        # Local optimization with L-BFGS-B on smooth objective
        try:
            res = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            v_lbfgs = res.x
            score_lbfgs = _exact_objective(v_lbfgs)
            
            if score_lbfgs < best_score:
                best_score = score_lbfgs
                best_v = v_lbfgs[:]
        except Exception as e:
            pass

    # Final polish with Nelder-Mead on the exact objective
    if best_v is not None and time.time() - t0 < TIME_LIMIT - 5.0:
        try:
            res = minimize(
                _exact_objective,
                best_v,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            v_nm = res.x
            score_nm = _exact_objective(v_nm)
            
            if score_nm < best_score:
                best_score = score_nm
                best_v = v_nm[:]
        except Exception:
            pass

    # If we still have time, try a few more random restarts from the best solution
    if best_v is not None and time.time() - t0 < TIME_LIMIT - 5.0:
        for _ in range(5):
            if time.time() - t0 > TIME_LIMIT - 2.0:
                break
            v_perturbed = best_v[:]
            # Perturb slightly
            for i in range(len(v_perturbed)):
                if i % 3 == 2: # Angle
                    v_perturbed[i] += random.gauss(0, 0.5)
                else:
                    v_perturbed[i] += random.gauss(0, 0.05)
            
            try:
                res = minimize(
                    _exact_objective,
                    v_perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                v_res = res.x
                score_res = _exact_objective(v_res)
                if score_res < best_score:
                    best_score = score_res
                    best_v = v_res[:]
            except Exception:
                pass

    # Extract result
    inner = []
    for i in range(N):
        inner.append([float(best_v[3*i]), float(best_v[3*i+1]), float(best_v[3*i+2])])
    
    center = (float(best_v[3*N]), float(best_v[3*N+1]))
    angle = float(best_v[3*N+2])
    
    return inner, center, float(best_score), angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
