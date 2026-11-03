import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_vertices(cx, cy, angle):
    """Get vertices of a unit regular hexagon centered at (cx,cy) with rotation angle."""
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return np.array(verts)

def _sat_separation(v1, v2):
    """Check if two hexagons are separated using SAT. Returns minimum separation distance (negative if overlapping)."""
    min_sep = float('inf')
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                # Distance between projections
                if proj1.max() <= proj2.min():
                    sep = proj2.min() - proj1.max()
                elif proj2.max() <= proj1.min():
                    sep = proj1.min() - proj2.max()
                else:
                    sep = 0.0
                min_sep = min(min_sep, sep)
    return min_sep

def _all_separated(verts_list):
    """Check if all hexagons are pairwise separated."""
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            sep = _sat_separation(verts_list[i], verts_list[j])
            if sep < -1e-9:
                return False
    return True

def _compute_outer_side(verts_list):
    """Compute the minimum side length of a regular hexagon that contains all vertices."""
    all_points = np.vstack(verts_list)
    
    def side_for_angle(theta):
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        dots = all_points @ normals.T
        max_proj = np.max(np.abs(dots), axis=0)
        return np.max(max_proj) * 2.0 / math.sqrt(3)
    
    # Coarse search over theta
    theta_coarse = np.linspace(0, math.pi / 3, 50, endpoint=False)
    sides = [side_for_angle(t) for t in theta_coarse]
    min_idx = np.argmin(sides)
    theta0 = theta_coarse[min_idx]
    
    # Fine search
    result = minimize(side_for_angle, theta0, bounds=[(theta0 - math.pi/30, theta0 + math.pi/30)], method='bounded')
    return result.fun

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Initial configuration: center + 6 ring 1 + 4 ring 2
    d = math.sqrt(3)
    centers_init = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers_init.append((d * math.cos(angle), d * math.sin(angle)))
    
    # Add 4 from ring 2 (alternating to avoid symmetry issues)
    ring2_indices = [0, 2, 4, 6]  # Every other position in ring 2
    for idx in ring2_indices:
        q, r = [(0,2),(1,1),(2,0),(2,-1),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-1,2)][idx]
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        centers_init.append((x, y))
    
    assert len(centers_init) == N
    
    # Objective function
    def objective(params):
        if time_left() < 5.0:
            return 1e10
        
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Build vertices
        verts_list = []
        for i in range(N):
            verts = _hex_vertices(xs[i], ys[i], thetas[i])
            verts_list.append(verts)
        
        # Check separation and compute penalty
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = _sat_separation(verts_list[i], verts_list[j])
                if sep < 0:
                    sep_penalty += sep * sep * 1000.0
        
        # Compute outer side length
        outer_side = _compute_outer_side(verts_list)
        
        return outer_side + sep_penalty
    
    # Initial parameters
    init_params = []
    for cx, cy in centers_init:
        init_params.extend([cx, cy, 0.0])
    init_params = np.array(init_params)
    
    best_result = None
    best_score = float('inf')
    
    # Try Nelder-Mead first
    if time_left() > 30.0:
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Try multiple random restarts with Nelder-Mead
    np.random.seed(42)
    restarts = 5
    for r in range(restarts):
        if time_left() < 20.0:
            break
        
        # Perturb initial configuration
        perturbed = init_params.copy()
        for i in range(N):
            perturbed[3*i] += np.random.normal(0, 0.3)
            perturbed[3*i+1] += np.random.normal(0, 0.3)
            perturbed[3*i+2] = np.random.uniform(-math.pi/6, math.pi/6)
        
        try:
            res = minimize(objective, perturbed, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # If we have a good result, verify and return
    if best_result is not None and time_left() > 5.0:
        xs = best_result[0::3]
        ys = best_result[1::3]
        thetas = best_result[2::3]
        
        # Final verification
        verts_list = [_hex_vertices(xs[i], ys[i], thetas[i]) for i in range(N)]
        
        # Check all separations
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                sep = _sat_separation(verts_list[i], verts_list[j])
                if sep < -1e-8:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            outer_side = _compute_outer_side(verts_list)
            inner = [[float(xs[i]), float(ys[i]), float(math.degrees(thetas[i]))] for i in range(N)]
            return inner, (0.0, 0.0), float(outer_side), 0.0
    
    # Fallback: return initial configuration
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers_init]
    verts_list = [_hex_vertices(cx, cy, 0.0) for cx, cy in centers_init]
    outer_side = _compute_outer_side(verts_list)
    return inner, (0.0, 0.0), float(outer_side), 0.0
