"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Rigid-Body RRT* search + Nelder-Mead refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin before 150s kill

# Unit hexagon vertices (pointy-top, side length 1)
# Vertices at angles 30, 90, 150, 210, 270, 330 degrees
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Edge normals (outward) for containment check
# For a regular hexagon centered at origin, the 6 supporting lines have normals at 0,60,...,300
EDGE_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def get_cluster_vertices(local_centers, global_pos, global_angle_deg):
    """
    Compute global vertices for a rigid cluster.
    local_centers: (N, 2) array of local positions (relative to cluster center)
    global_pos: (2,) array
    global_angle_deg: float
    Returns: (N, 6, 2) array of global vertices
    """
    a_rad = np.radians(global_angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate local centers
    rotated_centers = local_centers @ R.T + global_pos
    
    # Generate vertices
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = UNIT_VERTS @ R.T + rotated_centers[i]
    return verts

def compute_required_side(verts, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain vertices.
    """
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    # Flatten vertices to (N*6, 2) for faster processing
    flat_verts = verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    max_extent = np.max(max_proj)
    
    # Apothem = side * sqrt(3)/2 => side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_cluster_validity(local_centers, global_pos, global_angle_deg, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """
    Check if the rigid cluster is valid:
    1. No internal overlaps (pre-computed for base layout, but check just in case if layout changes)
    2. Containment in outer hexagon
    """
    verts = get_cluster_vertices(local_centers, global_pos, global_angle_deg)
    
    # 1. Check Internal Overlaps (SAT)
    # Since we use a fixed valid base layout, this should always pass.
    # But we include it for robustness.
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            axes = axes / (norms + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check Containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    flat_verts = verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    
    if np.max(max_proj) > apothem + tol:
        return False
    
    return True

def generate_base_layout():
    """
    Generate a compact, non-overlapping local layout for 12 hexagons.
    Uses a central hexagon + 6 neighbors + 5 from the next ring.
    This layout is rigid.
    """
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # Hex coordinates for a compact cluster
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: Pick 5 to make 12 total.
    # A known compact packing for 12 is a 3x3 grid minus corners? Or hex cluster.
    # Let's use the hex cluster: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    # Wait, (2,-1) is distance sqrt(3^2 + 1.5^2)? No.
    # Let's just list valid hex coords that don't overlap.
    # Distinct hexes in axial coords (q, r) must have hex distance >= 1.
    # Actually, for side length 1, center-to-center distance must be >= sqrt(3).
    
    coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), # 7
        (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0) # 5 more
    ]
    
    # Verify no overlaps in this specific set
    # (2,0) is dist 2*sqrt(3) from (0,0). OK.
    # (2,-1) is dist sqrt(3^2 + (-1.5)^2)? No, axial to cartesian:
    # x = sqrt(3)*(q + r/2), y = 1.5*r
    # (2,-1): x = sqrt(3)*(2 - 0.5) = 1.5*sqrt(3), y = -1.5.
    # Dist to (1,0) [x=sqrt(3), y=0]: dx = 0.5*sqrt(3), dy = -1.5. dist^2 = 0.75 + 2.25 = 3. dist = sqrt(3). OK (touching).
    # Dist to (0,0): dx = 1.5*sqrt(3), dy = -1.5. dist^2 = 6.75 + 2.25 = 9. dist = 3. OK.
    
    local_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        local_centers.append([x, y])
    
    local_centers = np.array(local_centers)
    
    # Center the local layout around origin to make global rotation/translation intuitive
    cx = np.mean(local_centers[:, 0])
    cy = np.mean(local_centers[:, 1])
    local_centers -= np.array([cx, cy])
    
    return local_centers

def objective_rigid(params, local_centers, outer_center, outer_angle_deg):
    """
    Objective function for Nelder-Mead on rigid body parameters.
    params: [x, y, angle_deg]
    """
    x, y, angle = params
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    side = compute_required_side(verts, outer_center, outer_angle_deg)
    
    # Penalty if containment is violated (should be 0 if side is computed correctly, 
    # but Nelder-Mead might explore weird spaces. We want to minimize side.
    # If side is the metric, it inherently satisfies containment.
    # However, we must ensure internal overlaps are not introduced. 
    # Since the layout is fixed and valid, internal overlaps are impossible.
    # So the objective is simply the required side length.
    return side

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # 1. Generate base rigid layout
    local_centers = generate_base_layout()
    
    # 2. RRT* style search over rigid body transformations
    # State: (x, y, angle)
    # We want to find (x, y, angle) that minimizes the required outer side length.
    # Note: The outer hexagon center is fixed at (0,0) and angle 0 for simplicity, 
    # or we can optimize it too. Let's fix outer center at (0,0) and angle 0, 
    # and optimize inner cluster pos/rot.
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_params = None
    best_side = float('inf')
    
    # Initial guess: centered at origin, angle 0
    current_params = np.array([0.0, 0.0, 0.0])
    
    # RRT* Expansion
    # We will perform random walks and keep the best node.
    # Since the space is continuous and small (3D), we can use a dense grid or random sampling + local refinement.
    # Given 120s budget, we can afford many evaluations.
    
    # Phase 1: Coarse Random Sampling
    num_samples = 2000
    for i in range(num_samples):
        if time.time() > DEADLINE - 30:
            break
        
        # Random perturbation around origin
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        angle = np.random.uniform(0, 60) # Hex symmetry
        
        params = np.array([x, y, angle])
        side = objective_rigid(params, local_centers, outer_center, outer_angle)
        
        if side < best_side:
            best_side = side
            best_params = params
    
    # Phase 2: Local Refinement using Nelder-Mead on the best candidates
    if best_params is not None:
        # Refine the best one
        res = minimize(
            objective_rigid, 
            best_params, 
            args=(local_centers, outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
        
        # Also try refining a few other good candidates from the sampling phase
        # (We didn't store them, so let's just do a few more random starts + refinement)
        
        for restart in range(5):
            if time.time() > DEADLINE - 10:
                break
            
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            angle = np.random.uniform(0, 60)
            params = np.array([x, y, angle])
            
            res = minimize(
                objective_rigid, 
                params, 
                args=(local_centers, outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
    
    # Final Validation and Construction
    if best_params is None:
        # Fallback
        best_params = np.array([0.0, 0.0, 0.0])
        best_side = objective_rigid(best_params, local_centers, outer_center, outer_angle)
    
    x, y, angle = best_params
    
    # Calculate final vertices to return
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    
    # Extract centers and angles for the return format
    # The return format expects: inner rows (x, y, angle_degrees)
    # Since the cluster is rigid, all inner hexagons have the same global angle.
    # The local positions are rotated by `angle` and translated by (x,y).
    
    a_rad = np.radians(angle)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    global_centers = local_centers @ R.T + np.array([x, y])
    
    inner_list = []
    for i in range(N):
        inner_list.append([global_centers[i, 0], global_centers[i, 1], angle])
    
    # Double check validity
    if not check_cluster_validity(local_centers, np.array([x, y]), angle, outer_center, best_side, outer_angle):
        # If invalid, try a safe fallback
        inner_list = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x_f = 3.5 * np.cos(theta)
            y_f = 3.5 * np.sin(theta)
            inner_list.append([x_f, y_f, 0.0])
        best_side = compute_required_side(get_cluster_vertices(np.array([[x_f, y_f] for x_f, y_f, _ in inner_list]), np.array([0,0]), 0), outer_center, outer_angle)
    
    return inner_list, (0.0, 0.0), best_side, outer_angle

if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
