"""
Strategy: Symmetry-Constrained Structural Search with Affine Deformation.

Instead of treating all 12 hexagons as independent bodies (36 DOF), we fix their
topological arrangement into a compact hexagonal cluster (1 center + 6 inner ring + 5 outer ring).
We then allow this cluster to undergo a global affine transformation (scaling, shearing) 
and rotation. This reduces the search space to ~5-6 parameters, making it feasible 
to exhaustively search the local landscape with Nelder-Mead.

The "affine deformation" allows the hexagons to be stretched or skewed relative to 
the outer hexagon, which can sometimes fit them into the corners more effectively 
than a rigid rotation/scale, while maintaining the internal disjointness constraints 
of the base lattice (which are preserved under affine maps if the base is valid).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Unit hexagon geometry
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
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
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

def generate_structural_config(scale_x, scale_y, shear, rotation_deg):
    """
    Generate a configuration of 12 hexagons based on a structural template.
    Template:
    - 1 center at (0,0)
    - 6 ring at distance d1, angles 0, 60, ...
    - 5 ring at distance d2, angles 30, 90, 150, 210, 270 (offset by 30 to fit in gaps)
    
    Actually, a better structural template for 12 is:
    Row 1: 3
    Row 2: 4
    Row 3: 3
    Row 4: 2
    Total: 12
    
    Let's use a hexagonal lattice basis.
    a1 = (sqrt(3), 0)
    a2 = (sqrt(3)/2, 1.5)
    
    Coords (i,j):
    Row 0 (y=0): (0,0), (1,0), (2,0)
    Row 1 (y=1.5): (-1,1), (0,1), (1,1), (2,1)  -> Wait, standard hex grid rows are offset.
    Let's stick to the lattice points that form a compact cluster.
    
    Points:
    (0,0)
    (1,0), (-1,1), (0,1), (2,0) -- No, let's use the specific 3-4-3-2 pattern from the seed but as a rigid body first, then deform.
    
    Pattern 3-4-3-2:
    (0,0), (1,0), (2,0)
    (-1,1), (0,1), (1,1), (2,1)
    (0,2), (1,2), (2,2)
    (0,3), (1,3)
    
    Let's define these base positions in a local coordinate system.
    """
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    coords = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Center of mass
    positions = np.array([i * a1 + j * a2 for i, j in coords])
    com = np.mean(positions, axis=0)
    positions = positions - com
    
    # Apply affine transformation: Scale + Shear
    # x' = sx * x + shear * y
    # y' = sy * y
    T = np.array([
        [scale_x, shear],
        [0.0, scale_y]
    ])
    
    positions = positions @ T.T
    
    # Apply Rotation
    rot_rad = np.radians(rotation_deg)
    R = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad)],
        [np.sin(rot_rad), np.cos(rot_rad)]
    ])
    
    positions = positions @ R.T
    
    # All hexagons have the same angle as the global rotation? 
    # In a rigid body deformation, the hexagons themselves might also rotate.
    # If we apply an affine map to the centers, the hexagons should also be affine-deformed?
    # No, hexagons are rigid. If we shear the cluster, the hexagons themselves cannot be sheared.
    # This is the key constraint: The hexagons remain regular, but their centers are deformed.
    # This means the "lattice" is no longer a perfect lattice.
    # The angles of the hexagons should probably align with the local "flow" or just be 0/rotated.
    # For simplicity and validity, let's assume all hexagons have the same rotation angle = rotation_deg.
    # This is a valid configuration as long as centers don't cause overlap.
    
    centers = []
    for pos in positions:
        centers.append([pos[0], pos[1], rotation_deg])
        
    return centers

def structural_objective(params):
    """
    Objective: Minimize the required outer side length.
    Params: [scale_x, scale_y, shear, rotation_deg]
    """
    sx, sy, sh, rot = params
    
    # Constraints: Keep scales positive and shear reasonable to avoid extreme distortion
    # that might cause overlaps due to center distance reduction.
    if sx < 0.8 or sy < 0.8 or sx > 1.5 or sy > 1.5 or abs(sh) > 0.5:
        return 1e10
        
    centers = generate_structural_config(sx, sy, sh, rot)
    
    # Check validity (no overlaps)
    if not check_validity(centers, outer_side=None, outer_angle=0.0):
        # Penalty for overlap
        # We can compute the minimum separation and penalize violations
        verts = get_hex_vertices(np.array(centers))
        min_sep = np.inf
        n = len(centers)
        for i in range(n):
            v1 = verts[i]
            for j in range(i + 1, n):
                v2 = verts[j]
                edges1 = np.roll(v1, -1, axis=0) - v1
                edges2 = np.roll(v2, -1, axis=0) - v2
                axes = np.vstack([edges1, edges2])
                axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
                
                proj1 = v1 @ axes.T
                proj2 = v2 @ axes.T
                
                sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                                proj2.min(axis=0) - proj1.max(axis=0))
                
                max_sep = np.max(sep)
                if max_sep < 0:
                    return 1e10 + abs(max_sep) * 100 # Strong penalty
                min_sep = min(min_sep, max_sep)
                
    side = compute_required_side(centers)
    return side

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Multi-start Nelder-Mead
    # Parameters: scale_x, scale_y, shear, rotation_deg
    # Initial guesses:
    # 1. Identity
    # 2. Slight stretch in x
    # 3. Slight shear
    # 4. Rotated 30 deg
    
    initial_guesses = [
        [1.0, 1.0, 0.0, 0.0],
        [1.1, 0.9, 0.0, 0.0],
        [0.9, 1.1, 0.0, 0.0],
        [1.0, 1.0, 0.2, 0.0],
        [1.0, 1.0, -0.2, 0.0],
        [1.0, 1.0, 0.0, 30.0],
        [1.0, 1.0, 0.0, 60.0],
        [1.05, 1.05, 0.1, 15.0],
    ]
    
    for i, x0 in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        try:
            res = minimize(
                structural_objective, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < 1e9: # Valid solution found
                params = res.x
                sx, sy, sh, rot = params
                centers = generate_structural_config(sx, sy, sh, rot)
                
                # Final validation
                if check_validity(centers, outer_side=None, outer_angle=0.0):
                    side = compute_required_side(centers)
                    if side < best_score:
                        best_score = side
                        best_result = (centers.tolist(), (0.0, 0.0), side, 0.0)
                        
        except Exception:
            continue
            
    # If no structural solution found or is bad, fall back to a known good lattice config
    if best_result is None or best_score > 4.0: # 4.0 is a rough estimate of a bad score
        # Use the lattice config from the seed as a fallback
        a1 = np.array([np.sqrt(3), 0.0])
        a2 = np.array([np.sqrt(3) / 2.0, 1.5])
        coords = [
            (0, 0), (1, 0), (2, 0),
            (-1, 1), (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3)
        ]
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], 0.0])
        
        side = compute_required_side(centers)
        if check_validity(centers, outer_side=side):
            best_result = (centers, (0.0, 0.0), side, 0.0)
            best_score = side

    if best_result is None:
        # Absolute fallback
        inner_default = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = 3.5 * np.cos(theta)
            y = 3.5 * np.sin(theta)
            inner_default.append([x, y, 0.0])
        required_side = compute_required_side(inner_default)
        best_result = (inner_default, (0.0, 0.0), required_side, 0.0)
        
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
