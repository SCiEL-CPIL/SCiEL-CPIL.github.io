import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12
SQRT3 = np.sqrt(3)

def hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return vertices of a regular hexagon with given center, side length, and rotation.
    Angle 0 corresponds to a pointy-top hexagon (vertex at angle 0? No, let's define it).
    Standard definition: Vertices at angles 0, 60, 120...
    If angle_deg=0, vertex 0 is at (side, 0).
    """
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def soft_max(values, beta):
    """
    Smooth approximation of max.
    """
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m > 100:
        return m
    shifted = values - m
    exp_vals = np.exp(beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return m + np.log(np.sum(exp_vals)) / beta

def soft_min(values, beta):
    """
    Smooth approximation of min.
    """
    return -soft_max(-values, beta)

def required_L_for_point(x, y):
    """
    Calculate the minimum side length L of a flat-top hexagon centered at origin
    that contains the point (x, y).
    Flat-top hexagon inequalities:
    1. |x| <= L
    2. |y| <= sqrt(3)/2 * L  => L >= 2/sqrt(3) * |y|
    3. |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L => L >= 2/sqrt(3) * |sqrt(3)/2 * x + 1/2 * y|
       => L >= sqrt(3)/3 * |sqrt(3) * x + y|
    """
    c1 = abs(x)
    c2 = (2.0 / SQRT3) * abs(y)
    c3 = (SQRT3 / 3.0) * abs(SQRT3 * x + y)
    return max(c1, c2, c3)

def sat_separation(v1, v2):
    """
    Calculate minimum separation between two convex polygons v1, v2 (arrays of 6x2 vertices).
    Positive if separated, negative if overlapping.
    """
    min_sep = float('inf')
    # Collect all edge normals from both polygons
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            elen = np.linalg.norm(edge)
            if elen < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / elen
            axes.append(n)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            
    return min_sep

def objective(params):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    """
    # Unpack
    centers = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        centers.append((x, y, a))
    
    # Compute vertices for all inner hexagons
    all_verts = []
    for c in centers:
        v = hexagon_vertices(c[0], c[1], 1.0, c[2])
        all_verts.append(v)
        
    # 1. Calculate L_max (smooth max of required L for all vertices)
    L_reqs = []
    for v in all_verts:
        for pt in v:
            L_reqs.append(required_L_for_point(pt[0], pt[1]))
            
    # Use soft_max to get a smooth approximation of the maximum L required
    beta_L = 20.0
    L_smooth = soft_max(L_reqs, beta_L)
    
    # 2. Calculate pairwise separations
    seps = []
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_separation(all_verts[i], all_verts[j])
            seps.append(sep)
            
    # We want the minimum separation to be >= 0.
    # If min_sep < 0, they overlap.
    # Use soft_min to get a smooth approximation of the minimum separation.
    beta_sep = 20.0
    min_sep_smooth = soft_min(seps, beta_sep)
    
    # Penalty for overlap
    # If min_sep_smooth < 0, we want to penalize.
    # soft_max(-min_sep_smooth, beta_pen) will be 0 if min_sep_smooth >= 0, and large if < 0.
    beta_pen = 20.0
    penalty = soft_max(-min_sep_smooth, beta_pen)
    
    # Total objective
    # We want to minimize L_smooth, but heavily penalize overlaps.
    # lambda is a weight.
    lam = 10.0
    return L_smooth + lam * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configurations
    def get_lattice_positions():
        positions = []
        # Center
        positions.append((0, 0, 0))
        # Ring 1
        dx = 2.0 * SQRT3
        dy = 3.0
        ring1 = [
            (dx, 0, 0), (-dx, 0, 0),
            (dx/2, dy*0.866, 0), (-dx/2, dy*0.866, 0),
            (dx/2, -dy*0.866, 0), (-dx/2, -dy*0.866, 0)
        ]
        positions.extend(ring1)
        # Ring 2 (5 more to make 12)
        # We need 5 more positions.
        # The lattice continues.
        # (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866), (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866), (0, dy*1.732)
        ring2 = [
            (3*dx/2, dy*0.866, 0), (-3*dx/2, dy*0.866, 0),
            (3*dx/2, -dy*0.866, 0), (-3*dx/2, -dy*0.866, 0),
            (0, dy*1.732, 0)
        ]
        positions.extend(ring2)
        return positions

    def get_random_positions():
        positions = []
        for _ in range(N):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            a = random.uniform(0, 360)
            positions.append((x, y, a))
        return positions

    initial_configs = [
        get_lattice_positions(),
        get_random_positions(),
        get_random_positions()
    ]
    
    best_L = float('inf')
    best_params = None
    
    for config in initial_configs:
        if time.time() > deadline - 10:
            break
            
        params0 = np.array([p[0] for p in config] + [p[1] for p in config] + [p[2] for p in config])
        
        # Bounds
        bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                              bounds=bounds, 
                              options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.fun < best_L:
                # Extract the actual L from the result
                centers = []
                for i in range(N):
                    centers.append((result.x[3*i], result.x[3*i+1], result.x[3*i+2]))
                
                # Recalculate exact L_max to report
                all_verts = []
                for c in centers:
                    v = hexagon_vertices(c[0], c[1], 1.0, c[2])
                    all_verts.append(v)
                    
                L_reqs = []
                for v in all_verts:
                    for pt in v:
                        L_reqs.append(required_L_for_point(pt[0], pt[1]))
                
                exact_L = max(L_reqs)
                
                # Check validity (no overlaps)
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        sep = sat_separation(all_verts[i], all_verts[j])
                        if sep < -1e-5:
                            valid = False
                            
                if valid and exact_L < best_L:
                    best_L = exact_L
                    best_params = result.x.copy()
                    
        except Exception as e:
            pass

    # If no valid configuration was found, return a safe fallback
    if best_params is None:
        # Use the lattice with a large L
        config = get_lattice_positions()
        best_params = np.array([p[0] for p in config] + [p[1] for p in config] + [p[2] for p in config])
        # Calculate L for this
        centers = []
        for i in range(N):
            centers.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        all_verts = []
        for c in centers:
            v = hexagon_vertices(c[0], c[1], 1.0, c[2])
            all_verts.append(v)
        L_reqs = []
        for v in all_verts:
            for pt in v:
                L_reqs.append(required_L_for_point(pt[0], pt[1]))
        best_L = max(L_reqs)

    # Construct return values
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a = float(best_params[3*i+2])
        inner.append([x, y, a])
        
    center = (0.0, 0.0)
    side = float(best_L)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
