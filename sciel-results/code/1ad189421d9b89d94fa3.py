"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_T0 = time.time()
_DEADLINE = 145.0

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = s * math.cos(math.radians(30))
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # We fix the rotation of all inner hexagons to 0 to simplify constraints.
    # This ensures that if center distance >= sqrt(3), they are disjoint.
    # We also fix the outer hexagon rotation to 0 WLOG.
    
    # Variables: x_0, y_0, ..., x_{n-1}, y_{n-1}, t
    # Total 2n + 1 variables
    
    def objective(p):
        t = p[2*n]
        return t

    # Constraints are of the form: g(p) >= 0
    
    def constraints(p):
        cons = []
        x = p[:2*n]
        t = p[2*n]
        
        # 1. Boundary constraints: All vertices of inner hexagons must be inside outer
        # Outer hexagon: center (0,0), side t, rot 0.
        # Inequalities for point (vx, vy) to be inside outer hex (side t, rot 0):
        # |vx| <= t
        # |vy| <= t * sqrt(3)/2
        # |sqrt(3)*vx + vy| <= t * sqrt(3)
        # |sqrt(3)*vx - vy| <= t * sqrt(3)
        
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            # Vertices of unit hexagon at (xi, yi) with rot 0
            # Vertices are at angles 0, 60, 120, 180, 240, 300
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            verts = [
                (xi + 1.0, yi),
                (xi + 0.5, yi + math.sqrt(3)/2),
                (xi - 0.5, yi + math.sqrt(3)/2),
                (xi - 1.0, yi),
                (xi - 0.5, yi - math.sqrt(3)/2),
                (xi + 0.5, yi - math.sqrt(3)/2)
            ]
            for vx, vy in verts:
                # Constraint: vx <= t
                cons.append(t - vx)
                # Constraint: -vx <= t
                cons.append(t + vx)
                # Constraint: vy <= t * sqrt(3)/2
                cons.append(t * math.sqrt(3)/2 - vy)
                # Constraint: -vy <= t * sqrt(3)/2
                cons.append(t * math.sqrt(3)/2 + vy)
                # Constraint: sqrt(3)*vx + vy <= t * sqrt(3)
                cons.append(t * math.sqrt(3) - (math.sqrt(3)*vx + vy))
                # Constraint: -sqrt(3)*vx + vy <= t * sqrt(3)
                cons.append(t * math.sqrt(3) - (-math.sqrt(3)*vx + vy))

        # 2. Separation constraints: Distance between centers >= sqrt(3) + eps
        # (dist_i - dist_j)^2 >= (sqrt(3) + eps)^2
        # dist_i^2 + dist_j^2 - 2*dist_i*dist_j - (sqrt(3)+eps)^2 >= 0
        # This is non-convex. SLSQP might struggle.
        # Alternative: Use penalty method with Nelder-Mead or just check validity after optimization.
        # Let's try to minimize t with SLSQP, but SLSQP requires differentiable constraints.
        # The separation constraint $d_{ij} \ge D_{min}$ can be written as $d_{ij} - D_{min} \ge 0$.
        # $d_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}$.
        
        D_min = math.sqrt(3) + 0.01
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = x[2*i], x[2*i+1]
                xj, yj = x[2*j], x[2*j+1]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                cons.append(dist - D_min)
                
        return cons

    # Initial guess: Hexagonal lattice
    # Center at 0,0
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 6 hexagons at distance 2*sqrt(3) (but we only need 4 more for total 11)
    # Let's place 1 at center, 6 at ring 1, and 4 at ring 2 positions.
    
    p0 = np.zeros(2*n + 1)
    # Center
    p0[0] = 0.0
    p0[1] = 0.0
    
    # Ring 1
    for k in range(6):
        a = math.radians(60.0 * k)
        p0[2*(1+k)] = math.sqrt(3) * math.cos(a)
        p0[2*(1+k)+1] = math.sqrt(3) * math.sin(a)
        
    # Ring 2 (positions 7, 8, 9, 10)
    # Place them at angles 0, 60, 120, 180 on radius 2*sqrt(3)?
    # Or just spread them out.
    for k in range(4):
        a = math.radians(60.0 * k)
        p0[2*(7+k)] = 2.0 * math.sqrt(3) * math.cos(a)
        p0[2*(7+k)+1] = 2.0 * math.sqrt(3) * math.sin(a)
        
    # Initial t: Estimate based on bounding box
    # Max x is approx 2*sqrt(3)*1 + 1 = 4.46
    # So t should be > 4.46
    p0[2*n] = 5.0

    best_p = p0.copy()
    best_val = p0[2*n]
    
    # Try to optimize
    try:
        # SLSQP can be slow. Let's try Nelder-Mead with a penalty for constraints?
        # Or just use SLSQP with maxiter limit.
        # Given the complexity, Nelder-Mead with a penalty objective is often more robust for packing.
        
        def penalized_objective(p):
            t = p[2*n]
            x = p[:2*n]
            pen = 0.0
            # Boundary penalty
            for i in range(n):
                xi, yi = x[2*i], x[2*i+1]
                verts = [
                    (xi + 1.0, yi),
                    (xi + 0.5, yi + math.sqrt(3)/2),
                    (xi - 0.5, yi + math.sqrt(3)/2),
                    (xi - 1.0, yi),
                    (xi - 0.5, yi - math.sqrt(3)/2),
                    (xi + 0.5, yi - math.sqrt(3)/2)
                ]
                for vx, vy in verts:
                    d1 = vx - t
                    d2 = -vx - t
                    d3 = vy - t * math.sqrt(3)/2
                    d4 = -vy - t * math.sqrt(3)/2
                    d5 = (math.sqrt(3)*vx + vy) - t * math.sqrt(3)
                    d6 = (-math.sqrt(3)*vx + vy) - t * math.sqrt(3)
                    max_d = max(d1, d2, d3, d4, d5, d6)
                    if max_d > 0:
                        pen += max_d * max_d * 1000.0
            
            # Separation penalty
            D_min = math.sqrt(3) + 0.01
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi = x[2*i], x[2*i+1]
                    xj, yj = x[2*j], x[2*j+1]
                    dx = xi - xj
                    dy = yi - yj
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < D_min:
                        pen += (D_min - dist) * (D_min - dist) * 1000.0
            
            return t + pen

        # Use Nelder-Mead for robustness
        res = minimize(penalized_objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_val:
            # Check if valid (penalty is 0)
            x_res = res.x[:2*n]
            t_res = res.x[2*n]
            
            valid = True
            # Check boundary
            for i in range(n):
                xi, yi = x_res[2*i], x_res[2*i+1]
                verts = [
                    (xi + 1.0, yi),
                    (xi + 0.5, yi + math.sqrt(3)/2),
                    (xi - 0.5, yi + math.sqrt(3)/2),
                    (xi - 1.0, yi),
                    (xi - 0.5, yi - math.sqrt(3)/2),
                    (xi + 0.5, yi - math.sqrt(3)/2)
                ]
                for vx, vy in verts:
                    if not _point_in_hex(vx, vy, 0, 0, t_res, 0):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # Check separation
                D_min = math.sqrt(3) + 0.001
                for i in range(n):
                    for j in range(i+1, n):
                        xi, yi = x_res[2*i], x_res[2*i+1]
                        xj, yj = x_res[2*j], x_res[2*j+1]
                        dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                        if dist < D_min:
                            valid = False
                            break
                    if not valid: break
                
            if valid:
                best_p = res.x.copy()
                best_val = t_res
            else:
                # If not valid, we can't use this result directly.
                # But we might have improved the layout.
                pass

    except Exception:
        pass

    # If optimization didn't find a valid improvement, use the initial guess if it's valid
    # Or just ensure the returned result is valid.
    
    x = best_p[:2*n]
    t = best_p[2*n]
    
    # Final safety check: If the current best_p is invalid, fallback to a known valid config
    # We can construct a simple valid config:
    # Place hexagons in a grid with spacing sqrt(3) * 1.1
    # This might be large, but valid.
    
    def is_valid_config(x, t):
        # Check boundary
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            verts = [
                (xi + 1.0, yi),
                (xi + 0.5, yi + math.sqrt(3)/2),
                (xi - 0.5, yi + math.sqrt(3)/2),
                (xi - 1.0, yi),
                (xi - 0.5, yi - math.sqrt(3)/2),
                (xi + 0.5, yi - math.sqrt(3)/2)
            ]
            for vx, vy in verts:
                if not _point_in_hex(vx, vy, 0, 0, t, 0):
                    return False
        
        # Check separation
        D_min = math.sqrt(3) + 0.001
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = x[2*i], x[2*i+1]
                xj, yj = x[2*j], x[2*j+1]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < D_min:
                    return False
        return True

    if not is_valid_config(x, t):
        # Fallback: Linear arrangement or Grid
        # Grid: 3x4 grid?
        # Spacing dx = sqrt(3) + 0.1, dy = sqrt(3) + 0.1
        # This will be a valid config, just maybe not optimal.
        x_fb = np.zeros(2*n)
        t_fb = 10.0
        # Simple circle packing fallback from knowledge:
        # Min dist 1.75
        # Just place them in a spiral or grid
        # Let's use a grid of 3x4
        dx = 1.8
        dy = 1.8
        idx = 0
        for r in range(4):
            for c in range(3):
                if idx >= n: break
                x_fb[2*idx] = (c - 1) * dx
                x_fb[2*idx+1] = (r - 1.5) * dy
                idx += 1
        # Calculate required t for this config
        max_x = max(abs(x_fb[2*i]) + 1.0 for i in range(n))
        max_y = max(abs(x_fb[2*i+1]) + math.sqrt(3)/2 for i in range(n))
        # t must satisfy t >= max_x and t*sqrt(3)/2 >= max_y
        t_fb = max(max_x, max_y * 2 / math.sqrt(3)) + 0.1
        x = x_fb
        t = t_fb

    inner = [[float(x[2*i]), float(x[2*i+1]), 0.0] for i in range(n)]
    return inner, (0.0, 0.0), float(t), 0.0
