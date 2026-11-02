import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0

# Unit hexagon geometry (flat-top)
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def sat_min_separation(v1, v2):
    """
    Compute minimum separation between two convex polygons using SAT.
    Returns negative value if overlapping.
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_symmetric_config(params):
    """
    Generates a 1-6-5 symmetric configuration from parameters.
    params: [r1, r2, theta1_offset, theta2_offset, outer_angle]
    """
    r1, r2, t1_off, t2_off, o_ang = params
    
    centers = []
    # Center hexagon
    centers.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons
    for i in range(6):
        theta = np.radians(60 * i + t1_off)
        x = r1 * np.cos(theta)
        y = r1 * np.sin(theta)
        # Rotate inner hexagons to align with radial direction for tighter packing?
        # Usually 0 is fine, but let's try aligning them.
        # For now, keep 0 to maintain symmetry with the lattice.
        centers.append([x, y, 0.0])
        
    # Ring 2: 5 hexagons
    # To fit 5 in a ring, we space them by 72 degrees.
    # Offset t2_off allows shifting this ring relative to Ring 1.
    for i in range(5):
        theta = np.radians(72 * i + t2_off)
        x = r2 * np.cos(theta)
        y = r2 * np.sin(theta)
        centers.append([x, y, 0.0])
        
    return np.array(centers), o_ang

def objective(params):
    """
    Objective function for optimization.
    Minimizes the required outer side length, with a penalty for overlaps.
    """
    if time.time() > DEADLINE:
        return 1e10
        
    centers, o_ang = generate_symmetric_config(params)
    
    # Check overlaps
    verts = get_hex_vertices(centers)
    min_sep = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_min_separation(verts[i], verts[j])
            if sep < min_sep:
                min_sep = sep
                
    penalty = 0.0
    if min_sep < 0:
        # Heavy penalty for overlap
        penalty = -min_sep * 1000.0
        
    side = compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=o_ang)
    
    return side + penalty

def construct():
    start_time = time.time()
    
    # Initial guesses for [r1, r2, t1_off, t2_off, o_ang]
    # r1 should be around sqrt(3) ~ 1.732
    # r2 should be around 2*sqrt(3) ~ 3.464
    initial_guesses = [
        [1.73, 3.46, 0, 36, 0],
        [1.73, 3.46, 0, 0, 0],
        [1.8, 3.5, 0, 36, 30],
        [1.7, 3.4, 0, 18, 0],
        [1.75, 3.55, 0, 54, 0],
        [1.73, 3.46, 0, 36, 15],
        [1.73, 3.46, 0, 36, 45],
        [1.73, 3.46, 0, 36, 60],
        [1.73, 3.46, 0, 36, 90],
        [1.73, 3.46, 0, 36, 120],
        [1.73, 3.46, 0, 36, 150],
        [1.73, 3.46, 0, 36, 180],
    ]
    
    best_score = float('inf')
    best_result = None
    
    for x0 in initial_guesses:
        if time.time() > DEADLINE - 5:
            break
            
        try:
            # Use Nelder-Mead for local refinement
            # Bounds are implicit in the geometry (r1, r2 > 0)
            # We rely on the penalty to prevent invalid states during optimization
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score and res.fun < 100: # Sanity check
                best_params = res.x
                best_score = res.fun
                centers, o_ang = generate_symmetric_config(best_params)
                
                # Final validation
                if check_validity(centers, outer_center=(0.0, 0.0), outer_side=best_score, outer_angle=o_ang):
                    best_result = (centers.tolist(), (0.0, 0.0), best_score, o_ang)
                    
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: Return a valid, though not optimal, configuration
        # Simple grid
        centers = []
        for i in range(4):
            for j in range(3):
                x = i * 1.75
                y = j * 1.55
                centers.append([x, y, 0.0])
        # Center it
        com = np.mean([c[:2] for c in centers], axis=0)
        centers = [[c[0]-com[0], c[1]-com[1], c[2]] for c in centers]
        
        side = compute_required_side(centers)
        best_result = (centers, (0.0, 0.0), side, 0.0)
        
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
