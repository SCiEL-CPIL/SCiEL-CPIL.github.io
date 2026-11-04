import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of unit hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + k * np.pi / 3.0
        verts.append((cx + np.cos(ang), cy + np.sin(ang)))
    return np.array(verts)

def _required_side_length(positions, angles, cont_angle_deg):
    """Compute minimum side length of regular hexagon (center at origin, rotation cont_angle_deg)
    that contains all vertices of all inner hexagons."""
    ca = np.radians(cont_angle_deg)
    # Face normals of container hexagon (outward), at angles ca + k*60
    normals = []
    for k in range(6):
        ang = ca + k * np.pi / 3.0
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    max_proj = 0.0
    for i in range(len(positions)):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, angles[i])
        for v in verts:
            for n in normals:
                proj = v[0] * n[0] + v[1] * n[1]
                if proj > max_proj:
                    max_proj = proj
    # For a regular hexagon with side s, the apothem (distance from center to face) is s*sqrt(3)/2
    # So we need max_proj <= s * sqrt(3)/2
    return max_proj / (np.sqrt(3) / 2.0)

def _overlap_penalty(positions, angles, min_dist=1.95):
    """Penalty if any two hexagon centers are too close (approximate non-overlap check)."""
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty

def construct():
    sqrt3 = np.sqrt(3)
    
    # Step 1: Build a spectral graph embedding for initial layout
    # Create a ring-like graph: 12 nodes arranged in a cycle with some chords
    # This gives a compact initial layout
    
    n = N
    # Adjacency: connect each node to its neighbors in a cycle, plus some chords
    # for a more compact shape. Use a "hexagonal" adjacency pattern.
    # Nodes 0-5 form outer ring, 6-11 form inner ring, with cross connections.
    
    W = np.zeros((n, n))
    # Outer ring: 0-1-2-3-4-5-0
    for i in range(6):
        W[i, (i + 1) % 6] = 1.0
        W[(i + 1) % 6, i] = 1.0
    # Inner ring: 6-7-8-9-10-11-6
    for i in range(6):
        idx = 6 + i
        W[idx, 6 + (i + 1) % 6] = 1.0
        W[6 + (i + 1) % 6, idx] = 1.0
    # Cross connections: outer i to inner i and inner (i+1)%6
    for i in range(6):
        W[i, 6 + i] = 0.8
        W[6 + i, i] = 0.8
        W[i, 6 + (i + 1) % 6] = 0.5
        W[6 + (i + 1) % 6, i] = 0.5
    
    # Compute Laplacian
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Use the two smallest non-trivial eigenvectors (skip the first which is constant)
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    # Skip index 0 (trivial eigenvector), take next two
    v1 = eigenvectors[:, idx[1]]
    v2 = eigenvectors[:, idx[2]]
    
    # Scale to reasonable size (hexagons should be ~2 apart center-to-center)
    scale = 2.0
    init_x = v1 * scale
    init_y = v2 * scale
    
    # Center the layout
    init_x -= np.mean(init_x)
    init_y -= np.mean(init_y)
    
    # Initial parameters: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cont_angle]
    x0 = np.zeros(n * 3 + 1)
    for i in range(n):
        x0[i * 3] = init_x[i]
        x0[i * 3 + 1] = init_y[i]
        x0[i * 3 + 2] = 0.0  # initial rotation
    x0[-1] = 0.0  # container angle
    
    def objective(params):
        positions = []
        angles = []
        for i in range(n):
            positions.append((params[i * 3], params[i * 3 + 1]))
            angles.append(params[i * 3 + 2])
        cont_angle = params[-1]
        
        s = _required_side_length(positions, angles, cont_angle)
        penalty = _overlap_penalty(positions, angles, min_dist=1.95)
        return s + 50.0 * penalty
    
    # Nelder-Mead optimization
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5,
                               'adaptive': True})
    
    best_params = result.x
    positions = []
    angles = []
    inner = []
    for i in range(n):
        px = best_params[i * 3]
        py = best_params[i * 3 + 1]
        a = best_params[i * 3 + 2]
        positions.append((px, py))
        angles.append(a)
        inner.append([px, py, a])
    
    cont_angle = best_params[-1]
    side = _required_side_length(positions, angles, cont_angle)
    
    return inner, (0.0, 0.0), side, cont_angle
# EVOLVE-BLOCK-END
