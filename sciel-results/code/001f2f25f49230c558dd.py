import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _min_outer_side(points, outer_angle_deg=0.0):
    """
    Given a set of points, find the minimum side length of a regular hexagon
    with given rotation that contains all points, centered optimally.
    We approximate by centering at the centroid.
    """
    centroid = np.mean(points, axis=0)
    angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = angle + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    
    # For each point, compute the maximum dot product with any normal
    # The required half-width in each normal direction
    rel = points - centroid
    dots = rel @ normals.T  # shape: (n_points, 6)
    max_dots = np.max(dots, axis=0)  # shape: (6,)
    
    # The hexagon side length s satisfies: max_dot = s * sqrt(3) / 2
    # So s = 2 * max_dot / sqrt(3)
    s = 2.0 * np.max(max_dots) / np.sqrt(3)
    return s, centroid

def construct():
    # Define 12 hexagons in a honeycomb cluster
    # Using axial coordinates, convert to Cartesian
    # Hexagonal lattice: distance between centers = sqrt(3) for side=1
    # Basis vectors: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = np.sqrt(3)
    
    # 12 positions in a compact honeycomb arrangement
    # Let's use a 3x4 grid in hex coords, then adjust
    # Axial coords (q, r) -> Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    positions = []
    # Row r=-1: q = -1, 0, 1, 2 (4 hexagons)
    for q in range(-1, 3):
        positions.append((q, -1))
    # Row r=0: q = -1, 0, 1, 2 (4 hexagons)
    for q in range(-1, 3):
        positions.append((q, 0))
    # Row r=1: q = -1, 0, 1, 2 (4 hexagons)
    for q in range(-1, 3):
        positions.append((q, 1))
    
    # Convert to Cartesian
    centers = []
    for q, r in positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Center the cluster
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    def objective(theta):
        """Compute outer hexagon side length for rotation theta."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotate all centers
        rotated_centers = []
        for x, y in centers:
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            rotated_centers.append((rx, ry))
        
        # Get all vertices of all 12 hexagons
        all_verts = []
        for x, y in rotated_centers:
            verts = _hex_vertices(x, y, 1.0, np.degrees(theta))
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Find min outer hexagon side length
        s, centroid = _min_outer_side(all_verts, outer_angle_deg=0.0)
        return s
    
    # Ternary search over theta in [0, pi/3) due to 60-degree symmetry
    result = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-8})
    
    best_theta = result.x
    best_s = result.fun
    
    # Reconstruct the solution
    cos_t = np.cos(best_theta)
    sin_t = np.sin(best_theta)
    
    inner = []
    for x, y in centers:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        inner.append([rx, ry, np.degrees(best_theta)])
    
    # Find optimal center
    all_verts = []
    for x, y in centers:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        verts = _hex_vertices(rx, ry, 1.0, np.degrees(best_theta))
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    _, outer_center = _min_outer_side(all_verts, outer_angle_deg=0.0)
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_s), 0.0
