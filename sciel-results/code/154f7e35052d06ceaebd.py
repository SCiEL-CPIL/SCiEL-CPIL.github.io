import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # Define geometry
    # Unit hexagon vertices (side 1)
    # Flat-top orientation: vertices at angles 0, 60, 120, 180, 240, 300
    # Radius (center to vertex) = 1
    # Inradius (center to edge) = sqrt(3)/2
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Define cluster positions
    # Triangular lattice spacing d = sqrt(3)
    d = np.sqrt(3)
    positions = [
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        [1.5*d, 0.5*d*np.sqrt(3)],
        [-1.5*d, 0.5*d*np.sqrt(3)],
        [1.5*d, -0.5*d*np.sqrt(3)],
        [-1.5*d, -0.5*d*np.sqrt(3)]
    ]
    positions = np.array(positions)

    # Normals for the outer hexagon (fixed at 0 degrees for calculation, we rotate the cluster)
    # Outer hexagon side S, inradius R_in = S * sqrt(3) / 2
    # Condition: |p . n_k| <= R_in
    # n_k are unit normals to the faces
    # For a flat-top hexagon (vertices at 0, 60...), faces are at 30, 90, 150, 210, 270, 330
    # Wait.
    # Vertex at 0: (1,0). Face opposite is at 180. Normal is (-1,0).
    # Let's use the standard definition:
    # If vertices are at angles $k \pi/3$, the face normals are at $\pi/6 + k \pi/3$.
    # Let's verify:
    # Vertices: (1,0), (0.5, .866)...
    # Edge between (1,0) and (0.5, .866). Midpoint (0.75, .433).
    # Vector from origin to midpoint: angle 30 deg.
    # Yes, normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.pi/6 + np.arange(6) * np.pi/3
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in normal_angles
    ])

    def calc_min_side(theta):
        # Rotate positions and vertices by theta
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        # Global positions of centers
        centers = positions @ rot_mat.T
        
        # Global vertices of all hexagons
        # We need to rotate the local hex_verts by theta as well
        local_verts_rot = hex_verts @ rot_mat.T
        
        # Create all global vertices: centers[:, None, :] + local_verts_rot[None, :, :]
        # Shape: (11, 6, 2)
        all_verts = centers[:, None, :] + local_verts_rot[None, :, :]
        
        # Flatten to (66, 2)
        all_verts = all_verts.reshape(-1, 2)
        
        # Compute projections onto normals
        # Shape: (66, 6)
        projections = np.abs(all_verts @ normals.T)
        
        # Max projection
        max_proj = np.max(projections)
        
        # Convert max_proj (which is distance to face) to Side Length
        # R_in = max_proj
        # S = 2 * R_in / sqrt(3)
        return 2 * max_proj / np.sqrt(3)

    # Optimize theta
    # Search range [0, pi/6] due to symmetry
    result = minimize_scalar(calc_min_side, bounds=(0, np.pi/6), method='bounded', options={'xatol': 1e-8})
    
    best_theta = result.x
    best_side = result.fun
    
    # Construct output
    c, s = np.cos(best_theta), np.sin(best_theta)
    rot_mat = np.array([[c, -s], [s, c]])
    final_centers = positions @ rot_mat.T
    
    inner = []
    for i in range(N):
        x, y = final_centers[i]
        # Inner hexagons are rotated by best_theta
        inner.append([float(x), float(y), float(np.degrees(best_theta))])
    
    center = (0.0, 0.0)
    side = float(best_side)
    angle = 0.0 # Outer hexagon is fixed at 0, cluster is rotated
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
