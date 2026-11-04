import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 12
TIME_LIMIT = 148.0
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _seg_seg_dist(p1, p2, p3, p4):
    """Compute minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    elif a < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    elif e < 1e-12:
        s = -np.dot(d1, r) / a
        s = max(0.0, min(1.0, s))
        return np.linalg.norm(s * d1 - r)
        
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    if denom < 1e-12:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, (b * t - c) / a))
    else:
        s = (b * f - c * e) / denom
        if s < 0.0 or s > 1.0:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            t = (b * s + f) / e
            t = max(0.0, min(1.0, t))
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
                
    return np.linalg.norm(s * d1 - r - t * d2)

def _polygons_min_dist(v1, v2):
    """Compute minimum distance between two polygons (0 if overlapping/touching)."""
    min_d = float('inf')
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
    return min_d

def _point_in_hexagon(p, hex_verts):
    """Check if point p is inside hexagon defined by hex_verts (CCW)."""
    # Use cross product sign check
    for i in range(6):
        a = hex_verts[i]
        b = hex_verts[(i+1)%6]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < 0:
            return False
    return True

def _point_hex_dist(p, hex_verts):
    """Compute distance from point p to hexagon boundary (negative if inside)."""
    if _point_in_hexagon(p, hex_verts):
        # Distance to nearest edge (negative)
        min_d = float('inf')
        for i in range(6):
            a = hex_verts[i]
            b = hex_verts[(i+1)%6]
            # Project p onto segment ab
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
        return -min_d
    else:
        # Distance to nearest edge (positive)
        min_d = float('inf')
        for i in range(6):
            a = hex_verts[i]
            b = hex_verts[(i+1)%6]
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
        return min_d

def _objective(params, outer_angle=0.0, k_rep=100.0, k_cont=100.0, eps=1e-3):
    """
    Objective function: Minimize outer side length + penalties.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return float('inf')
        
    centers = params[0:2*N:3]
    angles = params[1:2*N+1:3]
    
    # Compute vertices for all inner hexagons
    inner_verts = []
    for i in range(N):
        v = _hex_vertices((centers[i], angles[i] if False else params[2*i+1]), 1.0, angles[i])
        # Wait, params structure: [x0, y0, a0, x1, y1, a1...]
        # So centers are 0, 3, 6... angles are 2, 5, 8...
        pass
    
    # Re-extract correctly
    centers = np.array([params[i] for i in range(0, 2*N, 3)])
    y_centers = np.array([params[i] for i in range(1, 2*N+1, 3)])
    angles = np.array([params[i] for i in range(2, 2*N+2, 3)])
    
    inner_verts = []
    for i in range(N):
        v = _hex_vertices((centers[i], y_centers[i]), 1.0, angles[i])
        inner_verts.append(v)
    
    # 1. Compute required outer side length (max projection)
    # Outer hexagon is centered at origin, rotated by outer_angle
    # Normals of outer hexagon
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        for v in inner_verts:
            projs = np.dot(v, normal)
            max_proj = max(max_proj, np.max(projs))
            
    # Side length s relates to apothem a = s * sqrt(3) / 2
    # Max projection from center is the apothem of the outer hexagon?
    # No, the outer hexagon is defined by its vertices.
    # If outer hexagon has side S, its apothem is S*sqrt(3)/2.
    # The condition "inside" means all inner vertices must satisfy:
    # dot(v, n_k) <= apothem_outer for all k.
    # So apothem_outer >= max_proj.
    # S = 2 * apothem_outer / sqrt(3).
    apothem_req = max_proj
    S = 2 * apothem_req / np.sqrt(3)
    
    # 2. Penalties
    penalty = 0.0
    
    # Collision penalty
    for i in range(N):
        for j in range(i+1, N):
            d = _polygons_min_dist(inner_verts[i], inner_verts[j])
            if d < eps:
                # Penalty grows as distance decreases
                # Use quadratic penalty for smoothness
                pen = (eps - d) ** 2 * k_rep
                penalty += pen
                
    # Containment penalty
    # We already calculated S based on max projection, so containment is "satisfied" by definition of S.
    # However, if we want to minimize S, we just minimize S.
    # But wait, the problem asks to minimize outer side length.
    # If we just return S, the optimizer will try to make S as small as possible.
    # The penalty for collision ensures they don't overlap.
    # Is there a penalty for being outside?
    # If we define S as the *required* size to contain them, then they are always contained in a hexagon of size S.
    # So the objective is simply S + collision_penalty.
    # This is a valid formulation: Minimize (Size needed to contain) subject to (No overlap).
    # If they overlap, Size might be small, but Penalty is huge.
    
    return S + penalty

def _objective_grad(params, outer_angle=0.0, k_rep=100.0, eps=1e-3):
    """Numerical gradient (or analytical if possible). Let's use numerical for robustness first."""
    # Finite difference gradient
    grad = np.zeros_like(params)
    h = 1e-6
    f0 = _objective(params, outer_angle, k_rep, eps)
    
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        f1 = _objective(params, outer_angle, k_rep, eps)
        params[i] = orig
        grad[i] = (f1 - f0) / h
        
    return grad

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Initial configurations
    np.random.seed(42)
    
    configs = []
    
    # 1. Hexagonal lattice (approximate)
    # Center at 0,0
    # 1st ring: 6 hexes at dist sqrt(3)
    # 2nd ring: 6 hexes at dist 2*sqrt(3) (but we need 12 total, so 1+6+5? or just scatter)
    # Let's use a simple grid or random scatter to start.
    
    # Random scatter
    for _ in range(5):
        pts = np.random.uniform(-3, 3, (N, 2))
        angs = np.zeros(N)
        params = np.zeros(3*N)
        for i in range(N):
            params[3*i] = pts[i,0]
            params[3*i+1] = pts[i,1]
            params[3*i+2] = angs[i]
        configs.append(params)
        
    # Lattice start
    pts = np.array([
        [0,0],
        [np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5],
        [2*np.sqrt(3), 0], [1.5*np.sqrt(3), 1.5], [0, 3],
        [-1.5*np.sqrt(3), 1.5], [-2*np.sqrt(3), 0]
    ])
    angs = np.zeros(N)
    params = np.zeros(3*N)
    for i in range(N):
        params[3*i] = pts[i,0]
        params[3*i+1] = pts[i,1]
        params[3*i+2] = angs[i]
    configs.append(params)
    
    # Optimization loop
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for init_params in configs:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Use L-BFGS-B
                # We don't have bounds strictly, but we can add them to prevent explosion
                # Bounds: x,y in [-10, 10], angle in [0, 60] (due to symmetry)
                bounds = []
                for i in range(N):
                    bounds.append((-10, 10)) # x
                    bounds.append((-10, 10)) # y
                    bounds.append((0, 60))   # angle (degrees)
                    
                res = minimize(
                    _objective, 
                    init_params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 1000}
                )
                
                if res.success or res.fun < best_side:
                    final_params = res.x
                    # Extract centers and angles
                    centers = np.array([final_params[i] for i in range(0, 2*N, 3)])
                    y_centers = np.array([final_params[i] for i in range(1, 2*N+1, 3)])
                    angles = np.array([final_params[i] for i in range(2, 2*N+2, 3)])
                    
                    # Calculate exact side
                    inner_verts = []
                    for i in range(N):
                        v = _hex_vertices((centers[i], y_centers[i]), 1.0, angles[i])
                        inner_verts.append(v)
                        
                    max_proj = 0.0
                    for k in range(6):
                        angle = np.deg2rad(outer_angle + k * 60)
                        normal = np.array([np.cos(angle), np.sin(angle)])
                        for v in inner_verts:
                            projs = np.dot(v, normal)
                            max_proj = max(max_proj, np.max(projs))
                            
                    S = 2 * max_proj / np.sqrt(3)
                    
                    # Verify validity (no overlap, inside)
                    is_valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            d = _polygons_min_dist(inner_verts[i], inner_verts[j])
                            if d < 1e-6: # Tolerance
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        if S < best_side:
                            best_side = S
                            best_inner = [[centers[i], y_centers[i], angles[i]] for i in range(N)]
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle
                            
            except Exception as e:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
