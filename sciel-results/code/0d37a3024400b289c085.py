"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

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

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
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

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _hex_radius(angle_deg, direction_deg):
    """
    Compute the radius of a unit hexagon in a specific direction.
    This is the distance from the center to the boundary along the direction.
    """
    a_rad = math.radians(angle_deg)
    d_rad = math.radians(direction_deg)
    dx_dir = math.cos(d_rad)
    dy_dir = math.sin(d_rad)
    
    # Hexagon vertices in local frame
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Edges are between these vertices.
    # The distance to the boundary is the minimum t > 0 such that the point (t*dx, t*dy) is on the boundary.
    # Alternatively, we can check the support function.
    # For a regular hexagon with side 1, the distance to the edge (apothem) is sqrt(3)/2.
    # The distance to a vertex is 1.
    
    # Let's use the support function approach: max dot product of vertex with direction? No, that's for convex hull.
    # Actually, the radius in direction d is min over edges of (apothem / cos(angle between d and edge normal)).
    # Or simply intersect the ray with the hexagon edges.
    
    # Simpler: The hexagon is defined by |x| <= 1, |y| <= sqrt(3)/2, |sqrt(3)x + y| <= 2 etc? No.
    # Let's stick to the geometric intersection.
    
    # Local frame vertices
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        verts.append((math.cos(theta), math.sin(theta)))
    
    # Rotate vertices to global orientation relative to the ray?
    # The ray is in global frame. The hexagon is rotated by angle_deg.
    # So we rotate the ray into the hexagon's local frame.
    ca, sa = math.cos(a_rad), math.sin(a_rad)
    lx = ca * dx_dir + sa * dy_dir
    ly = -sa * dx_dir + ca * dy_dir
    
    # Find intersection of ray (t * (lx, ly)) with hexagon edges in local frame.
    min_t = float('inf')
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        
        # Edge vector
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        
        # Ray: P(t) = t * (lx, ly)
        # Edge: Q(u) = v1 + u * (ex, ey), 0 <= u <= 1
        # Solve t * (lx, ly) = v1 + u * (ex, ey)
        # t * lx - u * ex = v1[0]
        # t * ly - u * ey = v1[1]
        
        det = -lx * ey + ly * ex
        if abs(det) < 1e-12:
            continue
            
        u = (-lx * v1[1] + ly * v1[0]) / det
        t = (ex * v1[1] - ey * v1[0]) / det
        
        if 0.0 <= u <= 1.0 and t > 1e-9:
            if t < min_t:
                min_t = t
                
    if min_t == float('inf'):
        # Should not happen for a convex polygon containing origin
        return 0.0
        
    return min_t

def _smooth_objective(v, penalty_weight=100.0, sigma=0.1):
    """
    Smooth objective for optimization.
    v = [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    # Outer side length
    s = _outer_hex_side(cfg, center, angle)
    
    # Collision penalty
    total_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            p1x, p1y, a1 = cfg[i]
            p2x, p2y, a2 = cfg[j]
            
            dx = p2x - p1x
            dy = p2y - p1y
            dist = math.hypot(dx, dy)
            
            if dist < 1e-9:
                dist = 1e-9
                dir_deg = 0.0
            else:
                dir_deg = math.degrees(math.atan2(dy, dx))
            
            # Distance from center 1 to boundary in direction of 2
            r1 = _hex_radius(a1, dir_deg)
            # Distance from center 2 to boundary in direction of 1 (opposite)
            r2 = _hex_radius(a2, dir_deg + 180)
            
            # Signed distance
            sep = dist - r1 - r2
            
            if sep < 0:
                # Overlapping. Use smooth penalty.
                # exp(-sep/sigma) is large when sep is very negative.
                # We want to penalize overlap.
                total_penalty += math.exp(-sep / sigma)
            elif sep < 2.0 * sigma:
                # Close but not overlapping. Mild penalty to encourage separation.
                total_penalty += math.exp(-sep / sigma) - 1.0
                
    return s + penalty_weight * total_penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
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
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    def make_cluster():
        """Tight cluster: central hex + ring of 6 + outer 5."""
        pts = []
        # Central
        pts.append([0.0, 0.0, 0.0])
        # Inner ring of 6
        R1 = 2.0 * math.sqrt(3) / 2.0
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        # Outer 5
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    # Initial best from heuristics
    inits = [make_hex_lattice(), make_ring(), make_cluster()]
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    for init in inits:
        s = _outer_hex_side(init, (0.0, 0.0), 0.0)
        if _is_valid(init):
            if s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = 0.0

    if best_cfg is None:
        # Fallback random
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-4, 4), random.uniform(-4, 4), 0.0])
        best_cfg = cfg
        best_score = 50.0

    # Differential Evolution Phase
    # Bounds for DE
    # x, y: -10 to 10
    # angle: 0 to 360
    bounds = [(-10, 10)] * N + [(0, 360)] * N + [(-10, 10)] * 2 + [(0, 360)]
    # Actually, structure is [x1,y1,a1, ... xN,yN,aN, cx, cy, cangle]
    # So:
    de_bounds = []
    for i in range(N):
        de_bounds.append((-10, 10)) # x
        de_bounds.append((-10, 10)) # y
        de_bounds.append((0, 360))  # angle
    de_bounds.append((-10, 10)) # cx
    de_bounds.append((-10, 10)) # cy
    de_bounds.append((0, 360))  # cangle
    
    # Time allocation
    de_time_budget = TIME_LIMIT - 20.0 # Leave 20s for polish
    
    try:
        # Run Differential Evolution
        # maxiter is not strictly time-bound, so we monitor time
        # We will use a callback to stop early if time is up? 
        # SciPy DE doesn't have a direct time limit, but we can wrap the objective.
        
        def timed_objective(v):
            if time.time() - t0 > de_time_budget:
                # Return a large value to signal stop? 
                # Actually, DE might not stop immediately.
                # We'll just let it run and check after.
                pass
            return _smooth_objective(v)

        # To be safe, we run DE with a reasonable maxiter and check time after.
        # If it takes too long, we will just take the best result so far.
        # However, DE is vectorized internally? No, it calls the function.
        # Let's just run it. If it exceeds time, we'll handle it.
        
        res = differential_evolution(
            _smooth_objective,
            de_bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=False # We will do our own polish
        )
        
        if time.time() - t0 < de_time_budget:
            v_de = res.x
            cfg_de = []
            for i in range(N):
                cfg_de.append([v_de[3*i], v_de[3*i+1], v_de[3*i+2]])
            center_de = [v_de[3*N], v_de[3*N+1]]
            angle_de = v_de[3*N+2]
            
            # Check validity and score
            if _is_valid(cfg_de):
                s_de = _outer_hex_side(cfg_de, tuple(center_de), angle_de)
                if s_de < best_score:
                    best_score = s_de
                    best_cfg = cfg_de
                    best_center = center_de
                    best_angle = angle_de
            else:
                # Try to repair or just skip if invalid
                # We can try a quick local polish to remove overlaps
                pass

    except Exception as e:
        # If DE fails, we still have best_cfg from heuristics
        pass

    # L-BFGS-B Polish Phase
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            # Flatten best_cfg
            v0 = []
            for p in best_cfg:
                v0.extend([p[0], p[1], p[2]])
            v0.extend([best_center[0], best_center[1], best_angle])
            
            # Use L-BFGS-B
            # We use a slightly lower penalty weight for polish to focus on geometry?
            # Or keep it high to ensure no overlap.
            # The smooth objective handles it.
            
            result = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() - t0 < TIME_LIMIT - 2.0:
                v_opt = result.x
                cfg_opt = []
                for i in range(N):
                    cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
                center_opt = [v_opt[3*N], v_opt[3*N+1]]
                angle_opt = v_opt[3*N+2]
                
                # Final validity check
                if _is_valid(cfg_opt):
                    s_opt = _outer_hex_side(cfg_opt, tuple(center_opt), angle_opt)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_cfg = cfg_opt
                        best_center = center_opt
                        best_angle = angle_opt
                else:
                    # If polish created overlap, try to fix by shrinking or moving?
                    # Or just revert to previous best if invalid.
                    # We can also try a Nelder-Mead polish on the invalid config to push out overlaps.
                    pass
        except Exception:
            pass

    # Final Nelder-Mead Polish if time remains and we want to squeeze more
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            def nm_objective(v):
                cfg = []
                for i in range(N):
                    cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
                center = (v[3*N], v[3*N+1])
                angle = v[3*N+2]
                
                if not _is_valid(cfg):
                    return 1e6 # Heavy penalty for invalid
                return _outer_hex_side(cfg, center, angle)
            
            v0 = []
            for p in best_cfg:
                v0.extend([p[0], p[1], p[2]])
            v0.extend([best_center[0], best_center[1], best_angle])
            
            # Nelder-Mead
            from scipy.optimize import minimize
            res_nm = minimize(
                nm_objective,
                v0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if time.time() - t0 < TIME_LIMIT - 1.0:
                v_nm = res_nm.x
                cfg_nm = []
                for i in range(N):
                    cfg_nm.append([v_nm[3*i], v_nm[3*i+1], v_nm[3*i+2]])
                center_nm = [v_nm[3*N], v_nm[3*N+1]]
                angle_nm = v_nm[3*N+2]
                
                if _is_valid(cfg_nm):
                    s_nm = _outer_hex_side(cfg_nm, tuple(center_nm), angle_nm)
                    if s_nm < best_score:
                        best_score = s_nm
                        best_cfg = cfg_nm
                        best_center = center_nm
                        best_angle = angle_nm
        except Exception:
            pass

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
