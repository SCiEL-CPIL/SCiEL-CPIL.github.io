"""Program for packing 12 unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.deg2rad(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _hexagon_overlap(inner1, inner2):
    """Check if two unit hexagons overlap using SAT."""
    x1, y1, a1 = inner1
    x2, y2, a2 = inner2
    
    v1 = _hex_vertices(x1, y1, 1.0, a1)
    v2 = _hex_vertices(x2, y2, 1.0, a2)
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
                axes.append(normal)
    
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def _construct_honeycomb_cluster():
    """Generate a honeycomb cluster of 12 unit hexagons, centered at origin."""
    sqrt3 = np.sqrt(3)
    centers = []
    for i in range(4):
        centers.append((i * 1.5, 0.0, 0.0))
    for i in range(4):
        centers.append((i * 1.5 + 0.75, sqrt3, 0.0))
    for i in range(4):
        centers.append((i * 1.5, 2 * sqrt3, 0.0))
    
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    centered = [(x - cx, y - cy, a) for (x, y, a) in centers]
    return centered

def _compute_min_outer_side(inner_positions, angle_offset=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    sqrt3 = np.sqrt(3)
    max_proj = 0.0
    
    base_angles = np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(base_angles + np.deg2rad(angle_offset)), 
                                 np.sin(base_angles + np.deg2rad(angle_offset))])
    
    for (x, y, a) in inner_positions:
        verts = _hex_vertices(x, y, 1.0, a)
        proj = verts @ normals.T
        max_proj = max(max_proj, np.max(proj))
    
    return max_proj * 2.0 / sqrt3

# EVOLVE-BLOCK-START
def construct():
    base_cluster = _construct_honeycomb_cluster()
    
    def objective(params):
        scale, angle_offset = params
        if scale <= 0:
            return 1e10
        
        scaled_inner = [(x * scale, y * scale, a) for (x, y, a) in base_cluster]
        side = _compute_min_outer_side(scaled_inner, angle_offset)
        
        overlap_penalty = 0.0
        for i in range(len(scaled_inner)):
            for j in range(i+1, len(scaled_inner)):
                if _hexagon_overlap(scaled_inner[i], scaled_inner[j]):
                    overlap_penalty += 1.0
        
        return side + 100.0 * overlap_penalty
    
    x0 = [1.0, 0.0]
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    scale_opt, angle_opt = result.x
    inner = [(x * scale_opt, y * scale_opt, a) for (x, y, a) in base_cluster]
    outer_center = (0.0, 0.0)
    outer_side = _compute_min_outer_side(inner, angle_opt)
    outer_angle = angle_opt
    inner_list = [[x, y, a] for (x, y, a) in inner]
    
    return inner_list, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
