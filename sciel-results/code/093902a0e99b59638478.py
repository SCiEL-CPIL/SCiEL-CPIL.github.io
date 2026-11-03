import numpy as np
import time
import itertools

START_TIME = time.time()
TIME_LIMIT = 145.0 # Safety margin for 150s limit

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    # Pointy top hexagon vertices
    # Angles: 90, 150, 210, 270, 330, 30
    angles = np.array([90, 150, 210, 270, 330, 30]) + angle_deg
    angles_rad = np.deg2rad(angles)
    x = cx + np.cos(angles_rad)
    y = cy + np.sin(angles_rad)
    return np.column_stack((x, y))

def min_enclosing_hex_side(vertices):
    # vertices: (M, 2) array
    # Returns (minimal_side, best_angle_deg)
    best_s = float('inf')
    best_alpha = 0.0
    # Symmetry of hexagon is 60 deg. 
    # We can search alpha in [0, 30) or [0, 60)
    # Let's try 60 steps for better precision
    for alpha_deg in np.linspace(0, 30, 60, endpoint=False):
        alpha_rad = np.deg2rad(alpha_deg)
        # Normals to sides of the outer hexagon
        # If vertices are at alpha + k*60, sides are at alpha + 30 + k*60
        # Let's define normals at angles alpha + 30 + k*60
        normal_angles = np.deg2rad(alpha_deg + 30 + np.arange(6)*60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        # Project all vertices onto normals
        # proj shape (M, 6)
        proj = vertices @ normals.T
        # Max absolute projection
        max_proj = np.max(np.abs(proj))
        # S * sqrt(3)/2 = max_proj => S = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_alpha = alpha_deg
    return best_s, best_alpha

def construct():
    # Define lattice configurations
    # Basis for triangular lattice of hexagons (pointy top)
    # Distance between centers: sqrt(3)
    # We scale by 1.01 to avoid floating point overlap issues
    
    scale = 1.01
    u = np.array([np.sqrt(3), 0]) * scale
    v = np.array([np.sqrt(3)/2, 1.5]) * scale
    
    def lattice_to_xy(i, j):
        return i*u + j*v

    # Generate candidate sets of indices
    # We want compact sets of N=11 points (as per the failed program's N, though not strictly required, it's a good baseline)
    # Or we can just try a few known good patterns.
    
    candidates = []
    
    # Pattern 1: Hexagonal number 7 + 4 more
    # Center (0,0), 6 neighbors
    hex7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    # Add 4 more to make 11
    # Neighbors of the ring: (2,0), (2,-1), (1,1), (-2,1) etc.
    # Let's just generate a few random-looking but compact sets
    
    # Set 1: Triangle 4-3-2-1 + 1
    idx1 = [(0,0), (1,0), (2,0), (3,0), 
            (0,1), (1,1), (2,1),
            (0,2), (1,2),
            (0,3),
            (3,1)]
            
    # Set 2: 4-4-3
    idx2 = [(0,0), (1,0), (2,0), (3,0),
            (0,1), (1,1), (2,1), (3,1),
            (0,2), (1,2), (2,2)]
            
    # Set 3: More circular
    # Center, 6 neighbors, 4 from next ring
    idx3 = [(0,0), 
            (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
            (2,0), (2,-1), (-2,1), (-2,2)] # 11 points
            
    # Set 4: Another variation
    idx4 = [(0,0), 
            (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
            (1,1), (-1,2), (2,-1), (-2,0)] # 11 points

    candidates = [idx1, idx2, idx3, idx4]
    
    best_s = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for idx in candidates:
        if time.time() > TIME_LIMIT:
            break
            
        centers = np.array([lattice_to_xy(i, j) for i, j in idx])
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        
        # Compute vertices for fixed inner orientation (0 deg)
        all_vertices = []
        for c in centers:
            verts = get_hex_vertices(c[0], c[1], 0.0)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        s, alpha = min_enclosing_hex_side(all_vertices)
        
        if s < best_s:
            best_s = s
            best_inner = [[c[0], c[1], 0.0] for c in centers]
            best_center = (0.0, 0.0)
            best_angle = alpha
            
    if best_inner is None:
        # Fallback: single hexagon
        best_inner = [[0.0, 0.0, 0.0]]
        best_center = (0.0, 0.0)
        best_s = 1.0
        best_angle = 0.0

    return best_inner, best_center, best_s, best_angle
