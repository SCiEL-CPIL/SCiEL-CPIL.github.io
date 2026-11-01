"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice basis
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    def hex_center_to_xy(i, j):
        return i * a1 + j * a2
    
    # Hex vertices for unit hexagon (pointy-top)
    hex_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    hex_verts_unit = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def get_hex_vertices(center, angle_deg=0.0):
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return hex_verts_unit @ R.T + center
    
    def enclosing_hex_side(centers, center):
        """Find min side length of regular hexagon containing all inner hexagons"""
        all_pts = []
        for c in centers:
            v = get_hex_vertices(c, 0.0)
            all_pts.append(v)
        all_pts = np.vstack(all_pts)
        
        best_s = float('inf')
        for rot_deg in range(0, 30):
            theta = np.radians(rot_deg)
            normal_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + theta
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            dots = (all_pts - center) @ normals.T
            max_dot = np.max(dots, axis=0)
            required_inradius = np.max(max_dot)
            s = required_inradius * 2 / np.sqrt(3)
            if s < best_s:
                best_s = s
        
        return best_s
    
    # Generate several candidate polyhex shapes for 11 cells
    # Shape 1: 3-4-3 arrangement (compact)
    shapes = []
    
    # Shape: hexagonal cluster of 19 cells, take subset
    # Better: manually define compact 11-cell polyhexes
    
    # Shape A: center + 6 neighbors + 4 more
    shape_a = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
               (2,-1), (2,0), (-2,1), (0,2)]
    
    # Shape B: 3-4-3 rows
    shape_b = [(0,0), (1,0), (2,0),
               (0,1), (1,1), (2,1), (3,1),
               (0,2), (1,2), (2,2), (3,2)]
    
    # Shape C: more compact
    shape_c = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
               (2,-1), (-2,1), (1,1), (-1,-1)]
    
    # Shape D: 4-3-4
    shape_d = [(0,0), (1,0), (2,0), (3,0),
               (0,1), (1,1), (2,1),
               (0,2), (1,2), (2,2), (3,2)]
    
    # Shape E: compact diamond
    shape_e = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
               (2,-1), (-2,1), (1,1), (-1,-1)]
    
    shapes = [shape_a, shape_b, shape_c, shape_d, shape_e]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for shape in shapes:
        if len(shape) != N:
            continue
        centers = np.array([hex_center_to_xy(i, j) for i, j in shape])
        com = np.mean(centers, axis=0)
        s = enclosing_hex_side(centers, com)
        if s < best_side:
            best_side = s
            inner = []
            for i, j in shape:
                cx, cy = hex_center_to_xy(i, j)
                inner.append([cx, cy, 0.0])
            best_inner = inner
            best_center = (com[0], com[1])
            best_angle = 0.0
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
