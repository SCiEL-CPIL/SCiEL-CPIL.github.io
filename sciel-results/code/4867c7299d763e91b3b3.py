import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
MIN_DIST = math.sqrt(3)  # Minimum distance between centers for non-overlapping same-orientation hexagons

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Vertices of a unit regular hexagon centered at origin, oriented with flat top (or pointy?)
    # Standard axial to cartesian: 
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0) ? No, let's use standard geometric definition.
    # Let's define vertices for a hexagon with side 1.
    # Angle 0 to 300 deg.
    # If we assume "flat-top" vs "pointy-top", it doesn't matter for same-orientation packing as long as consistent.
    # Let's use vertices at angles 0, 60, ..., 300.
    angles = np.linspace(0, 2 * math.pi, 7)[:-1]
    vertices = np.array([[math.cos(a), math.sin(a)] for a in angles])
    
    # Precompute vertex offsets
    # Note: The distance from center to vertex is 1.0.
    # The distance from center to edge (apothem) is sqrt(3)/2.
    
    def get_min_side_length(points, center, rot_deg):
        """
        Calculate the minimum side length of an outer hexagon (centered at 'center', rotated by 'rot_deg')
        that contains all inner hexagons defined by 'points' (centers) with same orientation.
        """
        rot_rad = math.radians(rot_deg)
        # Normals of the outer hexagon
        # The outer hexagon is defined by 6 half-planes.
        # The normals are at angles rot_rad + pi/6 + k*pi/3
        normals = []
        for k in range(6):
            ang = rot_rad + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            # For each inner hexagon, find the max projection of its vertices
            # proj = (p + v) . n = p . n + v . n
            # We need max over all i and v
            for p in points:
                px, py = p
                for vx, vy in vertices:
                    proj = (px + vx) * nx + (py + vy) * ny
                    # We consider distance from center
                    rel_proj = proj - (center[0] * nx + center[1] * ny)
                    if rel_proj > max_proj:
                        max_proj = rel_proj
                    if -rel_proj > max_proj:
                        max_proj = -rel_proj
        
        # max_proj is the distance from outer center to the furthest vertex of any inner hexagon
        # This distance must be <= R_out (inradius of outer hexagon)
        # R_out = S * sqrt(3) / 2
        # So S * sqrt(3) / 2 >= max_proj
        # S >= 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / math.sqrt(3)
        return S

    def optimize_outer(points):
        """
        Given fixed inner centers, find best outer center and rotation to minimize side length.
        """
        # Initial guess: center of mass, rotation 0
        cx0 = float(np.mean(points[:, 0]))
        cy0 = float(np.mean(points[:, 1]))
        
        # Grid search for rotation
        best_S = float('inf')
        best_center = (cx0, cy0)
        best_rot = 0.0
        
        # Coarse rotation search
        for i in range(37):
            rot = i * (math.pi / 6) / 36
            # For this rotation, the optimal center is the center of the minimal bounding hexagon.
            # This is a linear programming problem, but we can approximate by center of mass 
            # or use a small local optimization.
            # Given time limits, let's assume center of mass is close, or refine with a few steps.
            
            # Let's refine center using a simple gradient-free method or just use center of mass.
            # Actually, for a fixed rotation, the minimal containing hexagon center is unique.
            # We can find it by checking the intersection of the 6 strips.
            # But simpler: use minimize on center for fixed rot.
            
            def obj_center(c):
                return get_min_side_length(points, c, math.degrees(rot))
            
            # Use a few steps of Nelder-Mead for center
            try:
                res = minimize(obj_center, [cx0, cy0], method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                S_val = res.fun
                if S_val < best_S:
                    best_S = S_val
                    best_center = (res.x[0], res.x[1])
                    best_rot = rot
            except:
                pass

        # Refine rotation around best_rot
        try:
            def obj_rot(params):
                rot = params[0]
                # Re-optimize center for this rot? That's expensive.
                # Instead, we can just evaluate with current best_center, or do a 2D search.
                # Let's do a 2D search on (cx, cy, rot) directly.
                return get_min_side_length(points, (params[0], params[1]), math.degrees(params[2]))
            
            # Start from best found so far
            init = [best_center[0], best_center[1], best_rot]
            res = minimize(obj_rot, init, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_rot = res.x[2]
        except:
            pass
            
        return best_S, best_center, best_rot

    def objective(params):
        """
        Objective for Differential Evolution.
        params: x1, y1, ..., x11, y11
        """
        points = np.array(params).reshape(N, 2)
        
        # Check overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < MIN_DIST:
                    penalty += (MIN_DIST - dist) ** 2
        
        if penalty > 1e-6:
            # If overlapping, return a large value
            # We still need to return a finite value for DE
            # Estimate S to provide gradient-like info?
            # Just return large penalty + current S
            S, _, _ = optimize_outer(points)
            return S + penalty * 1000.0

        # No overlap, find optimal outer hexagon
        S, _, _ = optimize_outer(points)
        return S

    # Differential Evolution bounds
    # The hexagons must fit in some reasonable box.
    # Max side length estimate: If we place them in a row, length is ~11.
    # Let's set bounds to [-10, 10]
    bounds = [(-10, 10)] * (2 * N)
    
    # Run DE
    best_result = None
    try:
        # Use a subset of population to save time if needed, but DE is robust.
        # maxiter 100, popsize 20
        res = differential_evolution(objective, bounds, maxiter=100, popsize=20, tol=1e-4, seed=42, polish=True)
        if res.success or res.fun < float('inf'):
            best_result = res.x
            best_S = res.fun
    except Exception as e:
        # Fallback to a simple lattice if DE fails
        pass

    if best_result is None:
        # Fallback: Simple lattice
        # 3x3 grid + 2 extra
        pts = []
        for q in range(-1, 2):
            for r in range(-1, 2):
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append([x, y])
        # Add 2 more
        pts.append([math.sqrt(3) * 2, 0])
        pts.append([0, math.sqrt(3) * 1.5 * 2])
        points = np.array(pts)
        S, center, rot = optimize_outer(points)
        best_result = points.flatten()
        best_S = S
    else:
        points = np.array(best_result).reshape(N, 2)
        S, center, rot = optimize_outer(points)
        best_S = S

    # Finalize
    cx, cy = center
    angle_deg = math.degrees(rot)
    
    inner = []
    for x, y in points:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center_out = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center_out, side, rotation
