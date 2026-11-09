# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(cx, cy, d, angle_deg):
    """Compute edge normals (unit vectors) for a hexagon."""
    verts = _hex_vertices(cx, cy, d, angle_deg)
    normals = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal pointing outward: rotate edge 90 deg clockwise
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def _projection_interval(verts, n):
    """Project vertices onto normal n, return (min, max)."""
    dots = verts @ n
    return dots.min(), dots.max()

def _are_separated(verts1, verts2, tol=1e-6):
    """Check if two convex hexagons are separated using SAT."""
    normals = []
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    for i in range(6):
        p1 = verts2[i]
        p2 = verts2[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    
    for n in normals:
        min1, max1 = _projection_interval(verts1, n)
        min2, max2 = _projection_interval(verts2, n)
        # Check if intervals are disjoint (gap > tol)
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _is_inside_hexagon(v, outer_verts):
    """Check if point v is inside or on the boundary of outer hexagon (CCW vertices)."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _compute_outer_hex_side(centers, angle_deg, tol=1e-8):
    """
    Given centers of unit hexagons, compute the minimum side length of a regular hexagon
    with given angle that contains all vertices of all inner hexagons.
    The outer hexagon is centered at the centroid of inner centers.
    """
    centroid = centers.mean(axis=0)
    # All vertices of all inner hexagons
    all_verts = []
    for c in centers:
        verts = _hex_vertices(c[0], c[1], 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a hexagon centered at centroid with angle angle_deg and side s,
    # the distance from center to each edge midpoint is s * sqrt(3) / 2
    # The distance from center to each vertex is s
    # A point p is inside if for each edge normal n_i, (p - centroid) . n_i <= s * sqrt(3) / 2
    
    a = np.radians(angle_deg)
    # Edge normals for outer hexagon (pointing inward)
    # Vertices at angles a, a+60, ..., a+300
    # Edge between vertex i and i+1 has normal at angle a + 30 + 60*i (inward)
    normals = []
    for i in range(6):
        normal_angle = a + np.radians(30 + 60 * i)
        normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
    normals = np.array(normals)
    
    # For each vertex p, compute required s such that (p - centroid) . (-n_i) <= s * sqrt(3)/2 for all i
    # i.e., s >= 2/sqrt(3) * max_i (centroid - p) . n_i  for each p
    # Actually: (p - centroid) . n_i <= s * sqrt(3)/2  =>  s >= 2/sqrt(3) * (p - centroid) . n_i
    # But we need this for all i and all p, so s >= max over all p, all i of 2/sqrt(3) * (p - centroid) . n_i
    
    rel = all_verts - centroid
    # For each normal, project all points
    max_proj = 0
    for n in normals:
        projs = rel @ n
        max_proj = max(max_proj, projs.max())
    
    s = 2.0 / np.sqrt(3.0) * max_proj
    return s

def _objective(params):
    """
    Objective: minimize outer hexagon side length.
    params: 24 values = 12 centers (x,y each) + outer angle (1 value) = 25
    Actually let's fix outer angle to 0 for simplicity, or include it.
    Let's include outer angle as last param.
    """
    centers = params[:24].reshape(12, 2)
    outer_angle = params[24]
    
    # Separation constraints
    verts_list = [_hex_vertices(c[0], c[1], 1.0, 0.0) for c in centers]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _are_separated(verts_list[i], verts_list[j], tol=1e-6):
                return 1e6  # Penalty for overlap
    
    s = _compute_outer_hex_side(centers, outer_angle)
    return s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses numerical optimization to minimize the outer hexagon side length.
    """
    # Initial guess: hexagonal grid arrangement
    # Centers in hexagonal grid with spacing sqrt(3)
    sqrt3 = np.sqrt(3)
    # Arrange in a pattern: rows of hexagons
    # Let's try a compact arrangement
    initial_centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [-sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [sqrt3/2, -1.5],
        [-sqrt3/2, -1.5],
        [3*sqrt3/2, 1.5],
        [-3*sqrt3/2, 1.5],
        [3*sqrt3/2, -1.5],
        [-3*sqrt3/2, -1.5],
        [0, 3.0],
    ])
    initial_params = np.concatenate([initial_centers.flatten(), [0.0]])
    
    # Try multiple starting points and local optimizations
    best_params = initial_params
    best_s = _objective(initial_params)
    
    # Try different initial configurations
    configs = []
    
    # Config 1: The above
    configs.append(initial_params.copy())
    
    # Config 2: 4-4-4 rows
    row_y = [1.5, 0, -1.5]
    centers2 = []
    for i, y in enumerate(row_y):
        for j in range(4):
            x = (j - 1.5) * sqrt3
            centers2.append([x, y])
    configs.append(np.concatenate([np.array(centers2).flatten(), [0.0]]))
    
    # Config 3: 3-6-3
    centers3 = []
    for j in range(3):
        centers3.append([(j - 1) * sqrt3, 1.5])
    for j in range(6):
        centers3.append([(j - 2.5) * sqrt3, 0])
    for j in range(3):
        centers3.append([(j - 1) * sqrt3, -1.5])
    configs.append(np.concatenate([np.array(centers3).flatten(), [0.0]]))
    
    # Config 4: Hexagonal cluster - 1 center, 6 ring, 5 outer
    centers4 = [[0, 0]]
    for k in range(6):
        angle = np.radians(60 * k)
        centers4.append([sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)])
    # 5 more in second ring, at angles 0, 120, 240, 60, 300 but farther
    for k in [0, 2, 4, 1, 3]:
        angle = np.radians(60 * k)
        centers4.append([2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)])
    configs.append(np.concatenate([np.array(centers4).flatten(), [0.0]]))
    
    for config in configs:
        # Local optimization
        result = minimize(_objective, config, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    centers = best_params[:24].reshape(12, 2)
    outer_angle = best_params[24]
    s = best_s
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # angle in degrees
    
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
