"""Fresh design: optimize 11 unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 145.0

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def hex_normal_directions(angle_deg):
    """Get the 6 outward normal directions of a hexagon with given rotation."""
    base = np.radians(30) + np.radians(angle_deg)
    return [base + np.radians(60 * i) for i in range(6)]

def sat_separated(v1, v2, eps=1e-8):
    """SAT check: return True if hexagons v1, v2 are separated (no overlap)."""
    all_verts = [v1, v2]
    for verts in all_verts:
        for i in range(6):
            # Edge from verts[i] to verts[(i+1)%6]
            edge = verts[(i+1) % 6] - verts[i]
            # Normal to this edge (outward)
            normal = np.array([-edge[1], edge[0]])
            # Ensure outward: check against center
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - verts[i]) < 0:
                normal = -normal
            
            # Project both polygons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 + eps < min2 or max2 + eps < min1:
                return True  # Separated on this axis
    
    return False  # No separating axis found, so they overlap

def all_disjoint(centers, angles, eps=1e-7):
    verts_list = [hex_vertices(centers[i,0], centers[i,1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if not sat_separated(verts_list[i], verts_list[j], eps):
                return False
    return True

def min_outer_side(centers, angles):
    all_verts = []
    for i in range(N):
        v = hex_vertices(centers[i,0], centers[i,1], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    max_proj = 0
    for i in range(6):
        theta = np.radians(60 * i + 30)
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = all_verts[:,0] * nx + all_verts[:,1] * ny
        max_proj = max(max_proj, float(np.max(proj)))
    
    return max_proj / (np.sqrt(3) / 2)

def objective(params):
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    verts_list = [hex_vertices(centers[i,0], centers[i,1], angles[i]) for i in range(N)]
    overlap_count = 0
    for i in range(N):
        for j in range(i+1, N):
            if not sat_separated(verts_list[i], verts_list[j], 1e-7):
                overlap_count += 1
    
    S = min_outer_side(centers, angles)
    return S + 50.0 * overlap_count

def initial_config(seed=0):
    rng = np.random.RandomState(seed)
    positions = []
    angles = []
    
    # Central
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 around
    for i in range(6):
        theta = np.radians(60 * i)
        d = np.sqrt(3)
        positions.append([d * np.cos(theta), d * np.sin(theta)])
        angles.append(0.0)
    
    # 4 more in gaps
    for i in [0, 2, 4, 1]:
        theta = np.radians(60 * i + 30)
        d = 2.5
        positions.append([d * np.cos(theta), d * np.sin(theta)])
        angles.append(0.0)
    
    for i in range(N):
        positions[i][0] += rng.uniform(-0.05, 0.05)
        positions[i][1] += rng.uniform(-0.05, 0.05)
        angles[i] += rng.uniform(-5, 5)
    
    return np.array(positions).flatten(), np.array(angles)

def solve():
    best_S = float('inf')
    best_params = None
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        centers, angles = initial_config(seed=restart)
        params = np.concatenate([centers, angles])
        
        try:
            result = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 1e-3})
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        centers = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:]
        inner = [[float(centers[i,0]), float(centers[i,1]), float(angles[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(best_S), 0.0
    
    centers, angles = initial_config(seed=0)
    inner = [[float(centers[i]), float(angles[i]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def construct():
    return solve()
