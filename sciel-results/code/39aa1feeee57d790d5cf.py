import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin for 120s limit

def get_hex_vertices(center, angle_deg, side=1.0):
    """
    Returns the 6 vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    """
    cx, cy = center
    rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon centered at origin, flat-top orientation (angle 0)
    # Vertices are at angles 0, 60, 120, 180, 240, 300?
    # Standard flat-top hexagon vertices:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Wait, side length 1. Distance from center to vertex is 1.
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = np.array([
        [np.cos(a) * side, np.sin(a) * side] for a in angles
    ])
    # Rotate
    cos_r = np.cos(rad)
    sin_r = np.sin(rad)
    rot = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    verts = verts @ rot.T
    # Translate
    verts += center
    return verts

def objective(params, n_inner=12):
    """
    params: [x1, y1, ..., xn, yn, phi_outer]
    Returns: Side length L + penalty for overlaps
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return float('inf')
        
    centers = params[:2*n_inner].reshape(n_inner, 2)
    phi_outer = params[-1]
    
    # Calculate required L
    # Outer hexagon normals
    # Flat-top outer hexagon normals are at 30, 90, 150, 210, 270, 330?
    # If vertices are at 0, 60..., then edges are perpendicular to 30, 90...
    # Let's stick to the vertex definition.
    # If outer hexagon has angle phi, its vertices are at phi + k*60.
    # Its edge normals are at phi + 30 + k*60.
    normal_angles = np.deg2rad(phi_outer + 30 + 60*np.arange(6))
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in normal_angles
    ])
    
    max_proj = 0.0
    for i in range(n_inner):
        # Assume inner hexagons are aligned with angle 0 for simplicity in this version
        # Rotating inner hexagons usually doesn't help for hexagonal packing
        verts = get_hex_vertices(centers[i], 0.0)
        # Project vertices onto normals
        projs = verts @ normals.T
        # We need the max projection in any normal direction
        # The hexagon contains the point if v . n <= d for all n
        # So we need d >= max(v . n)
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
            
    # L = 2/sqrt(3) * d
    L = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Penalty for overlaps
    # Min distance between centers of disjoint unit hexagons is sqrt(3)
    penalty = 0.0
    min_dist = np.sqrt(3.0)
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_dist:
                penalty += (min_dist - dist)**2
                
    return L + 100.0 * penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_score = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. Hexagonal grid cluster
    # 2. Random jittered grid
    
    def generate_initial():
        # Create a 3x4 grid of points, then jitter
        # Hex grid spacing: sqrt(3)
        dx = np.sqrt(3)
        dy = 1.5 # Vertical spacing for hex grid is 1.5 if horizontal is sqrt(3)?
        # Let's just use a square grid with spacing sqrt(3) for simplicity of generation
        # Then optimizer will fix it
        pts = []
        # 3 cols, 4 rows
        for r in range(4):
            for c in range(3):
                x = c * dx + (0.5 * dx if r % 2 else 0)
                y = r * 1.5
                # Add some random jitter to break symmetry
                x += random.uniform(-0.1, 0.1)
                y += random.uniform(-0.1, 0.1)
                pts.append([x, y])
        # Center the cluster
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        pts = [[p[0]-cx, p[1]-cy] for p in pts]
        
        params = [coord for p in pts for coord in p]
        params.append(random.uniform(0, 60)) # Outer angle
        return np.array(params)

    # Run multiple optimizations
    for restart in range(50):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        x0 = generate_initial()
        
        # Use Powell method for robustness
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
                # Check if valid (no overlaps)
                centers = best_params[:2*N].reshape(N, 2)
                min_dist = np.sqrt(3.0)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(centers[i] - centers[j]) < min_dist - 1e-3:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Extract L from best_params
                    # Recalculate L without penalty
                    phi_outer = best_params[-1]
                    normal_angles = np.deg2rad(phi_outer + 30 + 60*np.arange(6))
                    normals = np.array([
                        [np.cos(a), np.sin(a)] for a in normal_angles
                    ])
                    max_proj = 0.0
                    for i in range(N):
                        verts = get_hex_vertices(centers[i], 0.0)
                        projs = verts @ normals.T
                        local_max = np.max(projs)
                        if local_max > max_proj:
                            max_proj = local_max
                    L = (2.0 / np.sqrt(3.0)) * max_proj
                    # Update best_score to actual L for tracking
                    best_score = L 
                    
        except Exception as e:
            continue

    if best_params is None:
        # Fallback: Large hexagon
        inner = [[i * np.sqrt(3), 0, 0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    centers = best_params[:2*N].reshape(N, 2)
    phi_outer = best_params[-1]
    
    # Calculate final L for score
    normal_angles = np.deg2rad(phi_outer + 30 + 60*np.arange(6))
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in normal_angles
    ])
    max_proj = 0.0
    for i in range(N):
        verts = get_hex_vertices(centers[i], 0.0)
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    L = (2.0 / np.sqrt(3.0)) * max_proj

    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    center = (0.0, 0.0)
    side = float(L)
    angle = float(phi_outer)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
