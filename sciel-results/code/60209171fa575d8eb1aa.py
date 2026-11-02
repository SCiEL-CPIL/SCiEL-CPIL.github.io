import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _check_valid(inner, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if packing is valid."""
    centers = [(inner[i][0], inner[i][1]) for i in range(N)]
    angles = [inner[i][2] for i in range(N)]
    
    # Check containment
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                return False
    
    # Check non-overlap using center distance (conservative)
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < np.sqrt(3) - 1e-6:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_side = float('inf')
    best_result = None
    
    # Try symmetric configurations first
    # Configuration 1: 3-4-3-2 arrangement (rows)
    sqrt3 = np.sqrt(3)
    
    # Generate several candidate symmetric layouts
    candidates = []
    
    # Layout: centered hexagonal arrangement
    # Place 12 hexagons in a pattern that fits well
    # Try: 4 rows of 3, with offsets
    for offset_x in [-0.5, 0.0, 0.5]:
        for offset_y in [-0.5, 0.0, 0.5]:
            inner = []
            positions = [
                (0, 0), (sqrt3, 0), (-sqrt3, 0),
                (sqrt3/2, 1.5), (-sqrt3/2, 1.5), (3*sqrt3/2, 1.5),
                (sqrt3/2, -1.5), (-sqrt3/2, -1.5), (3*sqrt3/2, -1.5),
                (0, 3), (sqrt3, 3), (-sqrt3, 3),
            ]
            for (x, y) in positions:
                inner.append([x + offset_x, y + offset_y, 0.0])
            candidates.append(inner)
    
    # Evaluate candidates
    for inner in candidates:
        # Find minimum outer side
        max_r = 0
        for pt in inner:
            r = np.sqrt(pt[0]**2 + pt[1]**2)
            max_r = max(max_r, r)
        
        # Binary search on outer side
        lo, hi = 2.0, 20.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if _check_valid(inner, 0, 0, mid, 0.0):
                hi = mid
            else:
                lo = mid
        
        if hi < best_side and _check_valid(inner, 0, 0, hi, 0.0):
            best_side = hi
            best_result = (inner, (0.0, 0.0), hi, 0.0)
    
    # Use optimization to refine
    elapsed = time.time() - start_time
    if elapsed < deadline - 20 and best_result is not None:
        def objective(params):
            if time.time() - start_time > deadline:
                return 1e6
            inner_params = params[:36]
            outer_cx = params[36]
            outer_cy = params[37]
            outer_side = params[38]
            outer_angle = params[39]
            
            if outer_side <= 0:
                return 1e6
            
            centers = []
            angles = []
            for i in range(N):
                x = inner_params[i*3]
                y = inner_params[i*3+1]
                ang = inner_params[i*3+2]
                centers.append((x, y))
                angles.append(ang)
            
            penalty = 0.0
            for i in range(N):
                verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                for v in verts:
                    if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                        penalty += 5.0
            
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
                    if dist < np.sqrt(3):
                        penalty += 5.0 * (np.sqrt(3) - dist)
            
            return outer_side + penalty
        
        # Start from best candidate
        inner0 = best_result[0]
        x0 = []
        for i in range(N):
            x0.extend([inner0[i][0], inner0[i][1], inner0[i][2]])
        x0.extend([best_result[1][0], best_result[1][1], best_result[2], best_result[3]])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_side:
                params = res.x
                inner = []
                for i in range(N):
                    inner.append([params[i*3], params[i*3+1], params[i*3+2]])
                ocx, ocy = params[36], params[37]
                oside, oangle = params[38], params[39]
                
                if _check_valid(inner, ocx, ocy, oside, oangle):
                    best_side = oside
                    best_result = (inner, (ocx, ocy), oside, oangle)
        except:
            pass
    
    if best_result is not None:
        return best_result
    
    # Ultimate fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

