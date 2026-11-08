import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hexagon_vertices(center, angle_deg):
    """
    Returns vertices of a unit regular hexagon centered at 'center' rotated by 'angle_deg'.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Unit hexagon vertices (side length 1)
    # Center at origin
    v = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    c_x, c_y = center
    # Rotation matrix
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate and translate
    rotated = v @ R.T
    translated = rotated + np.array([c_x, c_y])
    return translated

def _objective(params, n_normals=6):
    """
    params: [x0, y0, a0, x1, y1, a1, ...]
    Returns: Max projection + Overlap Penalty
    """
    # Unpack
    centers = np.array(params[0::3])
    angles = np.array(params[1::3])
    rotations = np.array(params[2::3])
    
    # Calculate vertices for all hexagons
    # Shape: (N, 6, 2)
    all_vertices = np.array([_hexagon_vertices((cx, cy), rot) for cx, cy, rot in zip(centers, angles, rotations)])
    
    # Normals of the outer hexagon (fixed orientation)
    # Outer hexagon is centered at (0,0) for the objective calculation of M
    # We assume the outer hexagon center is (0,0) and we minimize the side length L
    # which is proportional to M = max projection.
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    normal_angles = np.arange(0, 360, 60) * np.pi / 180.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Calculate projections
    # all_vertices: (N, 6, 2)
    # normals: (6, 2)
    # We need max |v . n|
    
    # Flatten vertices to (N*6, 2)
    verts_flat = all_vertices.reshape(-1, 2)
    
    # Projections: (N*6, 6)
    projections = verts_flat @ normals.T
    abs_projections = np.abs(projections)
    
    M = np.max(abs_projections)
    
    # Overlap Penalty
    # dist >= sqrt(3)
    # centers: (N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    # Penalty for dist < sqrt(3)
    # Use squared penalty for smoothness
    violations = np.maximum(0, np.sqrt(3) - dists)
    overlap_penalty = np.sum(violations**2)
    
    # Total objective
    return M + 10.0 * overlap_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Lattice cluster
    # 1 center + 6 neighbors + 4 more
    # Lattice vectors
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate a set of 11 lattice points
    # Center (0,0)
    # Neighbors (q, r) in {(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)} ?
    # Axial to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    # 7 central points
    pts = [(0,0)]
    # 6 neighbors
    neighbors = [
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    ]
    for q,r in neighbors:
        pts.append((q,r))
        
    # 4 more points from the next ring
    # Candidates: (2,0), (2,-1), (1,-2), (-1,-1) etc.
    # Let's pick 4 that are symmetric or compact.
    # (2,0) -> x=2sqrt(3), y=0
    # (1,-2) -> x=sqrt(3)*(1-1)=0, y=-3
    # (-1,-1) -> x=sqrt(3)*(-1-0.5)=-1.5sqrt(3), y=-1.5
    # (0,-2) -> x=sqrt(3)*(-1)=-sqrt(3), y=-3
    
    # Let's just add 4 arbitrary ones from the second ring and let the optimizer fix it.
    extra_candidates = [
        (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0),
        (-1,2), (0,2), (1,1), (2,1), (2,2), (1,3)
    ]
    # Pick 4
    extra = extra_candidates[:4]
    for q,r in extra:
        pts.append((q,r))
        
    # Convert to Cartesian
    init_centers = np.array([axial_to_cartesian(q, r) for q, r in pts])
    
    # Flatten params: x0, y0, a0, x1, y1, a1 ...
    # Initial angles 0
    init_params = np.zeros(N * 3)
    for i in range(N):
        init_params[i*3] = init_centers[i, 0]
        init_params[i*3+1] = init_centers[i, 1]
        init_params[i*3+2] = 0.0
        
    best_score = np.inf
    best_params = init_params.copy()
    
    # Try a few random restarts if time permits
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_params = init_params.copy()
        else:
            # Add small noise
            current_params = init_params + np.random.normal(0, 0.5, N*3)
            # Keep rotations in [0, 360)
            current_params[2::3] = current_params[2::3] % 360.0
            
        # Optimize
        try:
            res = minimize(
                _objective, 
                current_params, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            score = res.fun
            if score < best_score:
                best_score = score
                best_params = res.x
                
        except Exception as e:
            continue
            
    # Extract best result
    centers = best_params[0::3]
    angles = best_params[1::3]
    rotations = best_params[2::3]
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([float(centers[i]), float(angles[i]), float(rotations[i])])
        
    # Calculate M to get side length L
    # M is max projection
    # L = 2/sqrt(3) * M
    # We need to return the outer hexagon center, side, angle
    # The optimizer assumed outer center at (0,0) and angle 0.
    # However, the optimal outer center might not be (0,0) if the cluster is shifted?
    # The objective minimized the max projection from (0,0).
    # This effectively centers the bounding box at (0,0).
    # So the outer hexagon center is (0,0).
    
    # Recalculate M for the best params to be precise
    all_vertices = np.array([_hexagon_vertices((cx, cy), rot) for cx, cy, rot in zip(centers, angles, rotations)])
    verts_flat = all_vertices.reshape(-1, 2)
    normal_angles = np.arange(0, 360, 60) * np.pi / 180.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = verts_flat @ normals.T
    M = np.max(np.abs(projections))
    
    L = (2.0 / np.sqrt(3.0)) * M
    
    return inner, (0.0, 0.0), float(L), 0.0

