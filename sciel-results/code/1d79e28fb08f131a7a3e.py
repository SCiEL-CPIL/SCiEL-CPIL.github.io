"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    # Hexagon vertices for a unit regular hexagon with given angle
    def hexagon_vertices(angle_deg, center=(0,0)):
        a = np.radians(angle_deg)
        cx, cy = center
        verts = []
        for i in range(6):
            theta = a + i * np.pi / 3
            verts.append((cx + np.cos(theta), cy + np.sin(theta)))
        return np.array(verts)
    
    # Check if a point is inside a regular hexagon (center, side, rotation)
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        # Transform point to hexagon local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        x_local = dx * np.cos(-a) - dy * np.sin(-a)
        y_local = dx * np.sin(-a) + dy * np.cos(-a)
        # Hexagon with side s: vertices at angles 0,60,...,300 degrees
        # The hexagon is defined by |x_local| <= s*cos(30deg)*... 
        # Actually, for a regular hexagon with side s, the constraints are:
        # |x| <= s * cos(pi/6) * ... 
        # Let me use the standard form: a regular hexagon centered at origin with 
        # one vertex on x-axis has vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
        # The half-width in x is s, half-height is s*sqrt(3)/2
        # But it's a hexagon, so the constraint is:
        # For each edge, the point must be on the inside.
        # Simpler: the hexagon is the intersection of 6 half-planes.
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
        # For a hexagon with circumradius s (side length s), inradius = s*sqrt(3)/2
        # The half-planes: for each direction theta_k = pi/6 + k*pi/3, 
        #   x_local * cos(theta_k) + y_local * sin(theta_k) <= s * sqrt(3)/2
        inradius = side * np.sqrt(3) / 2
        for k in range(6):
            theta_k = np.pi/6 + k * np.pi/3
            if x_local * np.cos(theta_k) + y_local * np.sin(theta_k) > inradius + 1e-9:
                return False
        return True
    
    # Get all vertices of inner hexagon i
    def get_inner_hex_verts(i, params):
        x, y, ang = params[3*i:3*i+3]
        return hexagon_vertices(ang, center=(x, y))
    
    # Compute max violation: for each inner hex vertex, check if inside outer hex
    # Return the maximum "distance outside" (positive if outside)
    def containment_violation(params, outer_center, outer_side, outer_angle):
        cx, cy = outer_center
        max_viol = 0.0
        inradius = outer_side * np.sqrt(3) / 2
        a = np.radians(outer_angle)
        for i in range(N):
            x, y, ang = params[3*i:3*i+3]
            verts = hexagon_vertices(ang, center=(x, y))
            for vx, vy in verts:
                dx = vx - cx
                dy = vy - cy
                x_local = dx * np.cos(-a) - dy * np.sin(-a)
                y_local = dx * np.sin(-a) + dy * np.cos(-a)
                for k in range(6):
                    theta_k = np.pi/6 + k * np.pi/3
                    val = x_local * np.cos(theta_k) + y_local * np.sin(theta_k)
                    if val > inradius:
                        viol = val - inradius
                        if viol > max_viol:
                            max_viol = viol
        return max_viol
    
    # Non-overlap: for each pair, check min distance between hexagons
    # Approximate by center distance minus 2*inradius for same orientation
    # But with rotation, use a conservative bound
    def overlap_violation(params):
        max_viol = 0.0
        for i in range(N):
            xi, yi, ai = params[3*i:3*i+3]
            for j in range(i+1, N):
                xj, yj, aj = params[3*j:3*j+3]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                # Minimum non-overlap distance for two unit hexagons
                # Worst case (aligned): sqrt(3)
                # Use a slightly conservative value
                min_dist = np.sqrt(3) - 1e-6
                if dist < min_dist:
                    viol = min_dist - dist
                    if viol > max_viol:
                        max_viol = viol
        return max_viol
    
    def objective(params, outer_center, outer_side, outer_angle):
        cv = containment_violation(params, outer_center, outer_side, outer_angle)
        ov = overlap_violation(params)
        # Penalize violations heavily
        return cv * 100.0 + ov * 100.0
    
    # Initial hexagonal lattice arrangement
    # Rows: 2, 3, 4, 3 (total 12)
    # Hex lattice spacing: sqrt(3) for aligned hexagons
    dx = np.sqrt(3)
    dy = 1.5  # vertical spacing for hex lattice
    
    positions = []
    # Row 0: 2 hexagons
    for j in range(2):
        x = (j - 0.5) * dx
        y = 0.0
        positions.append((x, y, 0.0))
    # Row 1: 3 hexagons
    for j in range(3):
        x = (j - 1.0) * dx
        y = dy
        positions.append((x, y, 0.0))
    # Row 2: 4 hexagons
    for j in range(4):
        x = (j - 1.5) * dx
        y = 2*dy
        positions.append((x, y, 0.0))
    # Row 3: 3 hexagons
    for j in range(3):
        x = (j - 1.0) * dx
        y = 3*dy
        positions.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_init = (min(xs) + max(xs)) / 2
    cy_init = (min(ys) + max(ys)) / 2
    positions = [(p[0]-cx_init, p[1]-cy_init, p[2]) for p in positions]
    
    # Estimate initial outer side
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) + 1.0 for p in positions)
    # For a hexagon, circumradius = side, so initial side ~ max_r
    outer_side_init = max_r + 0.5
    
    # Build initial params
    params0 = []
    for p in positions:
        params0.extend([p[0], p[1], p[2]])
    
    # Optimize inner positions to fit in a fixed outer hexagon
    # Then optimize outer hexagon
    best_side = outer_side_init
    best_params = params0[:]
    
    # Phase 1: Optimize inner positions for a given outer hexagon
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    def obj_inner(params):
        if time.time() > deadline - 5.0:
            return 1e10
        return objective(params, outer_center, best_side, outer_angle)
    
    res = minimize(obj_inner, params0, method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-8})
    
    if res.fun < 1e-6:
        best_params = res.x
        # Try to shrink outer side
        # Binary search on side length
        lo, hi = 0.5, best_side
        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            mid = (lo + hi) / 2
            # Check if we can fit with side = mid
            # Re-optimize inner positions
            def obj_test(params):
                return objective(params, outer_center, mid, outer_angle)
            
            res2 = minimize(obj_test, best_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res2.fun < 1e-6:
                hi = mid
                best_params = res2.x
            else:
                lo = mid
        
        best_side = hi
    
    # Phase 2: Optimize outer angle and center
    def full_params(x):
        # x[0:3] = outer center x, y, angle
        # x[3:] = inner params
        ocx, ocy, oang = x[0], x[1], np.degrees(x[2])
        inner = x[3:]
        return inner, (ocx, ocy), best_side, oang
    
    # Optimize outer parameters
    x0 = [0.0, 0.0, 0.0] + best_params
    
    def obj_full(x):
        if time.time() > deadline - 5.0:
            return 1e10
        ocx, ocy, oang_rad = x[0], x[1], x[2]
        oang = np.degrees(oang_rad)
        inner = x[3:]
        cv = containment_violation(inner, (ocx, ocy), best_side, oang)
        ov = overlap_violation(inner)
        return cv * 100.0 + ov * 100.0
    
    res_full = minimize(obj_full, x0, method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-8})
    
    if res_full.fun < 1e-6:
        best_params = res_full.x[3:]
        outer_center = (res_full.x[0], res_full.x[1])
        outer_angle = np.degrees(res_full.x[2])
        
        # Try to shrink side again
        lo, hi = 0.5, best_side
        for _ in range(20):
            if time.time() > deadline - 10.0:
                break
            mid = (lo + hi) / 2
            def obj_test2(x):
                ocx, ocy, oang_rad = x[0], x[1], x[2]
                oang = np.degrees(oang_rad)
                inner = x[3:]
                cv = containment_violation(inner, (ocx, ocy), mid, oang)
                ov = overlap_violation(inner)
                return cv * 100.0 + ov * 100.0
            
            res2 = minimize(obj_test2, res_full.x, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res2.fun < 1e-6:
                hi = mid
                best_params = res2.x[3:]
                outer_center = (res2.x[0], res2.x[1])
                outer_angle = np.degrees(res2.x[2])
            else:
                lo = mid
        
        best_side = hi
    
    # Build result
    inner = []
    for i in range(N):
        x, y, ang = best_params[3*i:3*i+3]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_side), float(outer_angle)
# EVOLVE-BLOCK-END
