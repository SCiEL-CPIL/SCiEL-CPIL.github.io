import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT. Returns (disjoint, min_sep)."""
    n1 = len(verts1)
    n2 = len(verts2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_sep = float('inf')
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Separation on this axis
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            return False, 0.0 # Overlap found
        
        if sep < min_sep:
            min_sep = sep
            
    return True, min_sep

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r / (np.sqrt(3) * best_cos)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            disjoint, sep = _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol)
            if not disjoint:
                return False
    
    return True

def _smooth_objective(vars, N):
    """
    Objective function for L-BFGS-B.
    vars: [x1, y1, th1, ..., xN, yN, thN, S]
    """
    # Unpack
    # Last element is S
    S = vars[-1]
    if S < 0.5: # Prevent negative/zero side
        S = 0.5
    
    # Extract positions
    positions = []
    for i in range(N):
        x = vars[i*3]
        y = vars[i*3 + 1]
        th = vars[i*3 + 2]
        positions.append([x, y, th])
        
    # Center of mass to (0,0) to reduce drift? 
    # Actually, let the optimizer find the center. 
    # But symmetry suggests centering helps. 
    # We can subtract COM from x,y in the objective calculation for symmetry breaking?
    # No, better to let S and positions be free, but add a small penalty for COM != 0?
    # Or just fix one hexagon to 0,0? 
    # For now, let's just compute as is.
    
    # 1. Overlap Penalty
    overlap_penalty = 0.0
    all_verts = [_hex_vertices(x, y, 1.0, th) for (x, y, th) in positions]
    
    for i in range(N):
        for j in range(i+1, N):
            disjoint, sep = _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-9)
            if not disjoint:
                # sep is 0.0 if overlapping in this implementation. 
                # We need the penetration depth. 
                # The current _hexagons_disjoint returns 0.0 on overlap. 
                # We need to modify it to return the max penetration (min negative separation).
                # Let's re-calculate min_sep for overlapping case.
                # In _hexagons_disjoint, if overlap, it returns immediately. 
                # We need a version that returns the minimum separation (which is negative if overlapping).
                pass
    
    # I need to update _hexagons_disjoint to return the minimum separation distance (can be negative).
    # Let's create a helper _min_separation.
    
    return 0.0 # Placeholder

# Let's rewrite _hexagons_disjoint to return min_separation
def _min_separation(verts1, verts2):
    """Returns minimum separation distance. Positive if disjoint, negative if overlapping."""
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_sep = float('inf')
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            # Overlap on this axis. Separation is negative.
            # The depth of overlap on this axis is min(max1, max2) - max(min1, min2)
            # But for SAT, if they overlap on ALL axes, they are intersecting.
            # The "separation" is effectively the minimum translation needed to separate.
            # For penalty, we can use the minimum overlap depth across axes? 
            # Or just return a large negative number? 
            # A common smooth surrogate is to sum the squared negative separations.
            # If they overlap, sep = - (overlap_depth).
            # Let's calculate the overlap depth on this axis.
            overlap = min(max1, max2) - max(min1, min2)
            sep = -overlap
            
        if sep < min_sep:
            min_sep = sep
            
    return min_sep

def _smooth_objective(vars, N):
    S = vars[-1]
    if S < 0.5:
        S = 0.5
    
    positions = []
    for i in range(N):
        x = vars[i*3]
        y = vars[i*3 + 1]
        th = vars[i*3 + 2]
        positions.append([x, y, th])
        
    # 1. Overlap Penalty
    overlap_penalty = 0.0
    all_verts = [_hex_vertices(x, y, 1.0, th) for (x, y, th) in positions]
    
    for i in range(N):
        for j in range(i+1, N):
            sep = _min_separation(all_verts[i], all_verts[j])
            if sep < 0:
                overlap_penalty += sep * sep
                
    # 2. Containment Penalty
    # Check all vertices against outer hexagon of side S centered at (0,0) angle 0
    contain_penalty = 0.0
    outer_verts = _hex_vertices(0.0, 0.0, S, 0.0)
    # Actually, we can just use _point_in_hex or calculate distance to boundary.
    # For smoothness, let's use the required side length logic.
    for (ix, iy, iangle) in positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0]
            dy = v[1]
            r = np.sqrt(dx*dx + dy*dy)
            if r < 1e-12:
                continue
            theta = np.arctan2(dy, dx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            if best_cos < 1e-9:
                required_side = 100.0
            else:
                required_side = 2.0 * r / (np.sqrt(3) * best_cos)
            
            if required_side > S:
                contain_penalty += (required_side - S) ** 2
                
    # 3. Regularization to keep COM at 0
    com_x = np.mean([p[0] for p in positions])
    com_y = np.mean([p[1] for p in positions])
    com_penalty = 0.1 * (com_x**2 + com_y**2)
    
    # Total Objective
    # We want to minimize S. 
    # If there are violations, we add penalties.
    # Weights:
    w_overlap = 10.0
    w_contain = 10.0
    
    return S + w_overlap * overlap_penalty + w_contain * contain_penalty + com_penalty

def _optimize_lattice(init_pos, outer_angle=0.0):
    """Run L-BFGS-B optimization."""
    # Initial vector
    x0 = []
    for p in init_pos:
        x0.extend([p[0], p[1], p[2]])
    # Initial S: estimate from bounding box
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in init_pos)
    S0 = max_r * 2.0 + 2.0
    
    x0.append(S0)
    
    bounds = []
    for i in range(N):
        bounds.append((-20, 20)) # x
        bounds.append((-20, 20)) # y
        bounds.append((0, 60))   # angle (degrees, effectively 0-60 due to symmetry)
    bounds.append((1.0, 20.0)) # S
    
    res = minimize(_smooth_objective, x0, args=(N,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
    
    if not res.success:
        # Still return the best found
        pass
        
    x_opt = res.x
    S_opt = x_opt[-1]
    
    positions = []
    for i in range(N):
        positions.append([x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]])
        
    return positions, (0.0, 0.0), S_opt, outer_angle

def _initial_lattice():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_random():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 60)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def construct():
    start_time = time.time()
    
    initializations = [
        _initial_lattice,
        _initial_random
    ]
    
    best_result = None
    best_side = float('inf')
    
    for init_func in initializations:
        if _remaining_time() < 10.0:
            break
        
        init_pos = init_func()
        
        # Try optimization
        try:
            result = _optimize_lattice(init_pos)
            inner, center, side, cangle = result
            
            # Verify and refine
            if _check_validity(inner, center, side, cangle, tol=1e-5):
                if side < best_side:
                    best_side = side
                    best_result = result
            else:
                # If invalid, try to expand S slightly
                side += 0.1
                if _check_validity(inner, center, side, cangle, tol=1e-5):
                    if side < best_side:
                        best_side = side
                        best_result = (inner, center, side, cangle)
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0
        return inner, center, side, angle
    
    return best_result
