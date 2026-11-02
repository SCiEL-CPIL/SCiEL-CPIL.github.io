import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute base vertices for a unit hexagon (flat-top, vertex at angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 330
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_cluster_positions(scale=1.0):
    """
    Generate positions for 12 hexagons in a compact honeycomb-like cluster.
    We use a central hexagon (1) + ring of 6 (7 total) + 5 more from the next ring.
    Coordinates in axial system (q, r):
    Center: (0,0)
    Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 hexes
    Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    We need 5 more. Let's pick a contiguous arc to maintain compactness.
    (2,0), (1,1), (0,2), (-1,2), (-2,1)
    """
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    pos = []
    for q, r in coords:
        # Axial to pixel (pointy-top or flat-top?)
        # Standard hex grid math:
        # x = size * sqrt(3) * (q + r/2)
        # y = size * 3/2 * r
        # This corresponds to pointy-top if we consider y vertical.
        # Let's verify orientation.
        # If q=1, r=0: x=sqrt(3), y=0.
        # If q=0, r=1: x=sqrt(3)/2, y=1.5.
        # Distance between (0,0) and (1,0) is sqrt(3).
        # Distance between (0,0) and (0,1) is sqrt(3/4 + 2.25) = sqrt(0.75+2.25)=sqrt(3).
        # So neighbors are at distance sqrt(3).
        # For unit hexagons (side 1), the distance between centers in a touching lattice is 2 * apothem = sqrt(3).
        # So this lattice of touching hexagons has centers separated by sqrt(3).
        # This is correct for side length 1 hexagons.
        
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pos.append([x * scale, y * scale])
        
    return np.array(pos)

def hexagon_vertices(cx, cy, angle_deg):
    """
    Returns (6, 2) array of vertices for a unit hexagon at (cx, cy) with rotation angle_deg.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = (R @ BASE_VERTS.T).T
    return verts + np.array([cx, cy])

def check_validity(positions, angles, tol=1e-5):
    """
    Check if all hexagons are pairwise disjoint using SAT.
    """
    vertices_list = []
    for i in range(N):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i])
        vertices_list.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            p1 = vertices_list[i]
            p2 = vertices_list[j]
            
            disjoint = False
            # Check edges of p1
            for i_sat in range(6):
                j_sat = (i_sat + 1) % 6
                edge = p1[j_sat] - p1[i_sat]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal /= norm
                proj1 = p1.dot(normal)
                proj2 = p2.dot(normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                # Check edges of p2
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p2[j_sat] - p2[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                if not disjoint:
                    return False
    return True

def required_outer_side(positions, angles):
    """
    Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
    that contains all unit hexagons at given positions and angles.
    """
    verts_flat = np.zeros((N * 6, 2))
    for i in range(N):
        verts_flat[i*6:(i+1)*6] = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
    
    x = verts_flat[:, 0]
    y = verts_flat[:, 1]
    
    # Linear forms for regular hexagon containment (flat-top outer hexagon)
    # Normals:
    # 30 deg: (sqrt3/2, 1/2) -> proj = x*sqrt3/2 + y/2
    # 90 deg: (0, 1)        -> proj = y
    # 150 deg: (-sqrt3/2, 1/2) -> proj = -x*sqrt3/2 + y/2
    # 210 deg: (-sqrt3/2, -1/2) -> proj = -x*sqrt3/2 - y/2
    # 270 deg: (0, -1)       -> proj = -y
    # 330 deg: (sqrt3/2, -1/2) -> proj = x*sqrt3/2 - y/2
    
    L1 = x * (SQRT3/2) + y * 0.5
    L2 = y
    L3 = -x * (SQRT3/2) + y * 0.5
    L4 = -x * (SQRT3/2) - y * 0.5
    L5 = -y
    L6 = x * (SQRT3/2) - y * 0.5
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    max_proj = np.max(all_L)
    return max_proj * 2.0 / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy:
    # 1. Define a base cluster (honeycomb subset).
    # 2. Parameterize by:
    #    - Global rotation theta
    #    - Scale factor s (initially 1.0, can be adjusted to fit tighter? No, hexagons are unit size. 
    #      Wait, the task says "unit regular hexagons". So their size is fixed.
    #      The "scale" in my previous thought was for the *positions* to allow gaps? 
    #      If we scale positions, we increase gaps, which allows the outer hexagon to be smaller? 
    #      No, if we scale positions up, the cluster gets bigger, requiring a BIGGER outer hexagon.
    #      If we scale positions down, hexagons overlap. 
    #      So the "scale" must be >= 1.0 for validity if the base cluster is touching.
    #      Actually, the base cluster is touching. So scale=1.0 is the minimum for that specific shape.
    #      However, we can *rotate* individual hexagons to create gaps and allow a more compact global shape?
    #      Or we can use a different base shape that isn't fully touching?
    #      Let's stick to the idea: The positions are fixed relative to each other in the lattice, 
    #      but we can apply a global rotation. 
    #      Additionally, we can allow small "jitter" translations to optimize the fit.
    
    # Let's define the parameters for the optimizer:
    # p[0]: global rotation angle (radians)
    # p[1:13]: small translation offsets for each hexagon (dx, dy for each? No, that's 24 params + 1 = 25)
    # That's still a lot. 
    # Let's try a simpler approach first: Just global rotation and translation of the whole cluster?
    # The outer hexagon is centered at (0,0) and has fixed orientation (angle 0).
    # So translating the cluster doesn't help if it's centered? 
    # Actually, the "center" of the cluster might not align with the "center" of the bounding hexagon.
    # So we should allow a global translation (tx, ty) and global rotation (theta).
    # And maybe individual rotations?
    
    # Let's try: 
    # Variables:
    # theta: global rotation
    # tx, ty: global translation
    # local_rots[12]: individual rotation angles (relative to global)
    # local_trans[12]: individual translation vectors (dx, dy) - this might be overkill.
    
    # Let's start with: Global Rotation, Global Translation, and Individual Rotations.
    # Why individual rotations? Because rotating a hexagon changes its bounding box relative to the lattice.
    
    def objective(params):
        theta = params[0]
        tx = params[1]
        ty = params[2]
        local_rots = params[3:3+N]
        
        # Generate base positions
        base_pos = get_cluster_positions(scale=1.0)
        
        # Apply global rotation
        c, s = np.cos(theta), np.sin(theta)
        R_global = np.array([[c, -s], [s, c]])
        rotated_pos = (base_pos @ R_global.T) + np.array([tx, ty])
        
        # Apply local rotations
        # The orientation of each hexagon is theta + local_rot
        # Note: The vertices are calculated relative to the hexagon's center.
        # The "angle" in hexagon_vertices is the absolute angle of the hexagon.
        
        positions = rotated_pos
        angles = (np.degrees(theta) + local_rots) % 360
        
        # Check validity
        # If invalid, return a large penalty
        if not check_validity(positions, angles):
            return 100.0
            
        # Calculate side length
        side = required_outer_side(positions, angles)
        return side

    # Initial guess
    # Start with theta=0, tx=0, ty=0, local_rots=0
    x0 = np.zeros(3 + N)
    
    # Use Nelder-Mead for local optimization
    # It doesn't require gradients and handles non-smoothness (validity jumps) somewhat okay if penalty is high.
    # But validity is a hard constraint. 
    # A better approach: 
    # 1. Grid search for theta (0 to 60 degrees, step 1 degree)
    # 2. For each theta, run a fast local optimization for tx, ty and local rotations.
    
    best_side = float('inf')
    best_params = None
    
    # Phase 1: Coarse Grid Search on Global Rotation
    # The system has 6-fold symmetry, so 0 to 60 degrees is enough.
    for theta_deg in np.arange(0, 60, 1.0):
        if time.time() > deadline - 10:
            break
            
        theta = np.radians(theta_deg)
        
        # For each theta, we need to find best tx, ty and local rotations.
        # Let's fix local rotations to 0 for the coarse search to save time.
        # Just optimize tx, ty.
        
        def obj_tx_ty(params):
            tx, ty = params
            base_pos = get_cluster_positions(scale=1.0)
            c, s = np.cos(theta), np.sin(theta)
            R_global = np.array([[c, -s], [s, c]])
            positions = (base_pos @ R_global.T) + np.array([tx, ty])
            angles = np.full(N, theta_deg)
            
            if not check_validity(positions, angles):
                return 100.0
            return required_outer_side(positions, angles)
        
        # Optimize tx, ty
        res = minimize(obj_tx_ty, [0, 0], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_side:
            best_side = res.fun
            # Store the best configuration
            tx, ty = res.x
            base_pos = get_cluster_positions(scale=1.0)
            c, s = np.cos(theta), np.sin(theta)
            R_global = np.array([[c, -s], [s, c]])
            best_positions = (base_pos @ R_global.T) + np.array([tx, ty])
            best_angles = np.full(N, theta_deg)
            best_params = (theta_deg, tx, ty)

    if best_params is None:
        # Fallback to a known valid configuration if search failed
        positions = get_cluster_positions()
        angles = np.zeros(N)
        # Ensure valid
        if not check_validity(positions, angles):
             # Scale up
             scale = 1.0
             while scale < 5.0:
                 pos_scaled = positions * scale
                 if check_validity(pos_scaled, angles):
                     positions = pos_scaled
                     break
                 scale += 0.1
        best_side = required_outer_side(positions, angles)
        best_positions = positions
        best_angles = angles
    else:
        # Phase 2: Fine-tune with Local Rotations
        # Use the best (theta, tx, ty) as starting point.
        # Optimize local rotations.
        
        theta_deg, tx, ty = best_params
        theta = np.radians(theta_deg)
        
        def obj_full(params):
            # params: local_rots (12 values)
            local_rots = params
            base_pos = get_cluster_positions(scale=1.0)
            c, s = np.cos(theta), np.sin(theta)
            R_global = np.array([[c, -s], [s, c]])
            positions = (base_pos @ R_global.T) + np.array([tx, ty])
            angles = (theta_deg + local_rots) % 360
            
            if not check_validity(positions, angles):
                return 100.0
            return required_outer_side(positions, angles)
        
        x0_local = np.zeros(N)
        res_local = minimize(obj_full, x0_local, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res_local.fun < best_side:
            best_side = res_local.fun
            local_rots = res_local.x
            base_pos = get_cluster_positions(scale=1.0)
            c, s = np.cos(theta), np.sin(theta)
            R_global = np.array([[c, -s], [s, c]])
            best_positions = (base_pos @ R_global.T) + np.array([tx, ty])
            best_angles = (theta_deg + local_rots) % 360

    # Final check and return
    if not check_validity(best_positions, best_angles):
        # Emergency fallback
        positions = get_cluster_positions()
        angles = np.zeros(N)
        scale = 1.0
        while scale < 5.0:
            pos_scaled = positions * scale
            if check_validity(pos_scaled, angles):
                best_positions = pos_scaled
                break
            scale += 0.1
        best_angles = angles
        best_side = required_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
