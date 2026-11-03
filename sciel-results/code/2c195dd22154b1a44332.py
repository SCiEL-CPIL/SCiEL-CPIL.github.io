"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    s3 = np.sqrt(3)
    
    # Hexagonal lattice basis for tangent unit hexagons
    a1 = np.array([s3, 0.0])
    a2 = np.array([s3/2, 1.5])
    
    # Unit hexagon vertices (side 1, centered at origin)
    unit_vertices = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
    
    def enclosing_hexagon_side(centers, angle_deg=0.0):
        """Compute minimal side length of outer regular hexagon containing all unit hexagons."""
        angle = np.radians(angle_deg)
        normals = []
        for k in range(6):
            theta = angle + k * np.pi / 3 + np.pi / 6
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        max_side = 0.0
        for n in normals:
            v_dots = unit_vertices @ n
            max_v = np.max(v_dots)
            max_proj = 0.0
            for c in centers:
                proj = np.dot(c, n) + max_v
                if proj > max_proj:
                    max_proj = proj
            required_s = max_proj * 2.0 / s3
            if required_s > max_side:
                max_side = required_s
        return max_side
    
    def optimize_angle(centers):
        best_side = float('inf')
        best_angle = 0.0
        for angle_deg in np.linspace(0, 60, 61):
            side = enclosing_hexagon_side(centers, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
        res = minimize_scalar(lambda a: enclosing_hexagon_side(centers, a), 
                              bounds=(best_angle - 5, best_angle + 5), method='bounded')
        return res.fun, res.x
    
    # Try multiple arrangements
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_angle = 0.0
    
    # Arrangement 1: 4x3 rectangular block in hex grid
    positions = []
    for i in range(4):
        for j in range(3):
            positions.append(i * a1 + j * a2)
    centers = np.array(positions)
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    side, angle = optimize_angle(centers)
    if side < best_overall_side:
        best_overall_side = side
        best_overall_inner = centers.copy()
        best_overall_angle = angle
    
    # Arrangement 2: Hexagonal cluster (center + ring + partial)
    # Axial coordinates: (0,0), and 11 more
    positions = []
    # Center
    positions.append(np.array([0.0, 0.0]))
    # First ring (6 hexagons)
    for k in range(6):
        angle_k = k * np.pi / 3
        pos = s3 * np.array([np.cos(angle_k), np.sin(angle_k)])
        positions.append(pos)
    # Second ring partial (5 more to make 12)
    for k in range(6):
        if k == 3:  # skip one to make 5
            continue
        angle_k = k * np.pi / 3
        # Second ring positions are at distance 2*s3 in direction, but offset
        # Actually, second ring in hex grid: 2*a1, 2*a2, etc.
        # Let me use axial coords properly
        pass
    
    # Simpler: just try a few grid arrangements
    # Arrangement 3: 3x4 block
    positions = []
    for i in range(3):
        for j in range(4):
            positions.append(i * a1 + j * a2)
    centers = np.array(positions)
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    side, angle = optimize_angle(centers)
    if side < best_overall_side:
        best_overall_side = side
        best_overall_inner = centers.copy()
        best_overall_angle = angle
    
    # Arrangement 4: Compact hexagonal shape
    # Use axial coordinates (q, r) and convert to Cartesian
    def axial_to_cartesian(q, r):
        x = s3 * (q + r / 2)
        y = 1.5 * r
        return np.array([x, y])
    
    # Try a compact set of 12 axial coordinates
    axial_coords = [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1),
        (-1, 0), (0, -1), (1, -1), (2, -1), (3, 0), (-1, 1)
    ]
    positions = [axial_to_cartesian(q, r) for q, r in axial_coords]
    centers = np.array(positions)
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    side, angle = optimize_angle(centers)
    if side < best_overall_side:
        best_overall_side = side
        best_overall_inner = centers.copy()
        best_overall_angle = angle
    
    # Convert to output format
    inner = []
    for c in best_overall_inner:
        inner.append([c[0], c[1], 0.0])
    
    return inner, [0.0, 0.0], best_overall_side, best_overall_angle
# EVOLVE-BLOCK-END
