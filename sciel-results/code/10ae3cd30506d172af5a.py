"""Packing 11 unit hexagons in minimum enclosing regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

N = 11

def hex_center_to_vertices(center, side=1.0, angle_deg=0.0):
    """Get the 6 vertices of a regular hexagon centered at given point."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([
        center[0] + side * np.cos(angles),
        center[1] + side * np.sin(angles)
    ])

def get_all_vertices(inner_hexes):
    """Get all vertices from all inner hexagons."""
    verts = []
    for x, y, ang in inner_hexes:
        v = hex_center_to_vertices((x, y), 1.0, ang)
        verts.append(v)
    return np.vstack(verts)

def min_enclosing_hexagon_side(vertices, outer_center, outer_angle_deg):
    """
    Compute minimum side length of regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given vertices.
    """
    theta = np.radians(outer_angle_deg)
    # Normals to the 6 sides of the hexagon
    normal_angles = theta + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    rel = vertices - np.array(outer_center)
    projections = rel @ normals.T
    max_proj = np.max(np.abs(projections), axis=0)
    
    apothem_required = np.max(max_proj)
    side_length = apothem_required / (np.sqrt(3) / 2)
    
    return side_length

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Place 11 hexagons: 1 center + 6 ring + 4 outer
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = np.radians(60 * i)
        centers.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    for i in [0, 2, 4, 6]:
        angle = np.radians(60 * i)
        centers.append((2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)))
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    all_verts = get_all_vertices(inner)
    outer_center = (float(np.mean(all_verts[:, 0])), float(np.mean(all_verts[:, 1])))
    
    def objective(theta_deg):
        return min_enclosing_hexagon_side(all_verts, outer_center, theta_deg)
    
    best_side = np.inf
    best_angle = 0.0
    
    for start in np.arange(0, 60, 5):
        res = minimize_scalar(objective, bounds=(float(start), float(start + 60)), method='bounded', options={'xatol': 1e-7})
        if res.fun < best_side:
            best_side = float(res.fun)
            best_angle = float(res.x)
    
    return inner, outer_center, best_side, best_angle
# EVOLVE-BLOCK-END
