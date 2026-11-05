import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def point_in_convex(p, poly):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_overlap_area(v1, v2):
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def get_seed_params():
        """
        Returns the analytic seed configuration as a flat array of 33 floats:
        [x1, y1, a1, x2, y2, a2, ...]
        """
        eps = 1e-9
        centers = []
        angles = []
        
        # Center
        centers.append((0.0, 0.0))
        angles.append(30.0)
        
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            angles.append(30.0)
            
        # Ring 2: 4 hexagons
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
            angles.append(30.0)
            
        params = np.zeros(3 * N)
        for i in range(N):
            params[3*i] = centers[i][0]
            params[3*i+1] = centers[i][1]
            params[3*i+2] = angles[i]
            
        return params

    def compute_outer_side(params):
        """
        Compute the minimum side length of a regular hexagon (fixed orientation)
        that contains all inner hexagons defined by params.
        Outer hexagon is centered at (0,0) with angle 0 (flat top/bottom? No, pointy top).
        Let's assume outer hexagon vertices are at angles 0, 60, ...
        Normals are at 30, 90, ...
        """
        apothem_factor = math.sqrt(3) / 2.0
        # Normal angles for a hexagon with vertices at 0, 60, ...
        # Side 0 is between vertex 0 and 1? No.
        # Vertex k is at angle k * 60.
        # Side k is between vertex k and k+1.
        # The normal to side k points outward at angle k*60 + 30.
        normal_angles = [math.radians(30 + k * 60) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        
        max_proj = 0.0
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a_deg = params[3*i+2]
            verts = hex_vertices(x, y, 1.0, a_deg)
            for vx, vy in verts:
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        return max_proj / apothem_factor

    def overlap_penalty(params, lambda_val=100.0):
        """
        Calculate a smooth penalty for overlaps between inner hexagons.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                xi = params[3*i]
                yi = params[3*i+1]
                ai = params[3*i+2]
                xj = params[3*j]
                yj = params[3*j+1]
                aj = params[3*j+2]
                
                vi = hex_vertices(xi, yi, 1.0, ai)
                vj = hex_vertices(xj, yj, 1.0, aj)
                
                area = polygon_overlap_area(vi, vj)
                if area > 1e-10:
                    # Smooth penalty: square of area
                    penalty += lambda_val * (area ** 2)
                    
        return penalty

    def objective(params):
        s = compute_outer_side(params)
        pen = overlap_penalty(params, 100.0)
        return s + pen

    # Initial parameters
    p0 = get_seed_params()
    
    # Optimization
    best_params = p0
    best_score = compute_outer_side(p0)
    
    # Check initial validity
    # (The seed is known to be valid)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective,
            p0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if time.time() < deadline:
            p_opt = res.x
            s_opt = compute_outer_side(p_opt)
            # Check for overlaps
            pen = overlap_penalty(p_opt, 1.0) # Use lambda=1 to check raw penalty
            if pen < 1e-8: # Effectively no overlap
                if s_opt < best_score:
                    best_score = s_opt
                    best_params = p_opt
    except Exception:
        pass

    # If the first optimization didn't improve much or failed, try a few random restarts
    # We only do this if we have time.
    if time.time() < deadline - 20:
        for restart in range(5):
            # Perturb the seed
            noise = np.random.normal(0, 0.1, size=3*N)
            # Don't perturb angles too much
            for i in range(N):
                noise[3*i+2] *= 0.1
            p_restart = p0 + noise
            
            try:
                res_r = minimize(
                    objective,
                    p_restart,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
                )
                
                if time.time() > deadline - 10:
                    break
                    
                p_r = res_r.x
                s_r = compute_outer_side(p_r)
                pen_r = overlap_penalty(p_r, 1.0)
                
                if pen_r < 1e-8 and s_r < best_score:
                    best_score = s_r
                    best_params = p_r
                    
            except Exception:
                continue

    # Construct final result
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    # Outer hexagon is centered at (0,0), side length is best_score, angle 0
    outer_center = (0.0, 0.0)
    outer_side = best_score
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
