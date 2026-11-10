# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _compute_side(centers, hex_verts, theta):
    """Compute the required outer hexagon side length for given centers and outer angle."""
    normals = np.array([[np.cos(theta + np.deg2rad(60 * i)), np.sin(theta + np.deg2rad(60 * i))] for i in range(6)])
    proj_centers = centers @ normals.T
    max_proj_centers = np.max(proj_centers, axis=0)
    proj_hex = hex_verts @ normals.T
    max_proj_hex = np.max(proj_hex, axis=0)
    h_total = max_proj_centers + max_proj_hex
    max_h = np.max(h_total)
    return (2.0 / np.sqrt(3.0)) * max_h

def _refine_theta(centers, hex_verts, theta0, n_steps=200):
    """Local refinement around theta0 using golden-section search."""
    best_theta = theta0
    best_side = _compute_side(centers, hex_verts, theta0)
    
    # Golden section search in [theta0 - pi/180, theta0 + pi/180]
    a = theta0 - np.pi / 180.0
    b = theta0 + np.pi / 180.0
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = _compute_side(centers, hex_verts, c)
    fd = _compute_side(centers, hex_verts, d)
    
    for _ in range(n_steps):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _compute_side(centers, hex_verts, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _compute_side(centers, hex_verts, d)
    
    best_theta = (a + b) / 2.0
    best_side = _compute_side(centers, hex_verts, best_theta)
    return best_theta, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Unit hexagon vertices (angle 0)
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Triangular lattice vectors for ring 1
    d = np.sqrt(3)
    v = np.array([[d * np.cos(np.deg2rad(60 * i)), d * np.sin(np.deg2rad(60 * i))] for i in range(6)])
    
    # Ring 2 vectors
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])
        ring2.append(v[i] + v[(i + 1) % 6])
    ring2 = np.array(ring2)
    
    # Base centers: Center + Ring 1
    base_centers = np.vstack([np.array([[0, 0]]), v])
    
    # Generate more candidate sets of 4 ring-2 points
    # Include symmetric and asymmetric configurations
    candidates = [
        [0, 3, 6, 9],    # 0, 90, 180, 270
        [0, 2, 6, 8],    # 0, 60, 180, 240
        [1, 4, 7, 10],   # 30, 120, 210, 300
        [0, 5, 6, 11],   # 0, 150, 180, 330
        [1, 3, 7, 9],    # 30, 90, 210, 270
        [2, 4, 8, 10],   # 60, 120, 240, 300
        # Additional asymmetric candidates
        [0, 1, 6, 9],    # 0, 30, 180, 270
        [0, 3, 6, 10],   # 0, 90, 180, 300
        [0, 2, 6, 11],   # 0, 60, 180, 330
        [1, 4, 7, 8],    # 30, 120, 210, 240
        [1, 3, 7, 10],   # 30, 90, 210, 300
        [2, 4, 8, 9],    # 60, 120, 240, 270
        [0, 1, 6, 10],   # 0, 30, 180, 300
        [0, 3, 6, 11],   # 0, 90, 180, 330
        [1, 2, 7, 9],    # 30, 60, 210, 270
        [1, 4, 7, 11],   # 30, 120, 210, 330
        [2, 5, 8, 10],   # 60, 150, 240, 300
        [0, 4, 6, 10],   # 0, 120, 180, 300
        [1, 5, 7, 9],    # 30, 150, 210, 270
        [2, 3, 8, 11],   # 60, 90, 240, 330
    ]
    
    best_side = float('inf')
    best_config = None
    
    for idxs in candidates:
        if time.time() > deadline - 30:
            break
            
        r2_pts = ring2[idxs]
        centers = np.vstack([base_centers, r2_pts])
        
        # Coarse scan
        thetas = np.linspace(0, np.pi/3, 720)
        
        min_side_for_this_cluster = float('inf')
        best_theta = 0
        
        for theta in thetas:
            if time.time() > deadline - 15:
                break
            side = _compute_side(centers, hex_verts, theta)
            if side < min_side_for_this_cluster:
                min_side_for_this_cluster = side
                best_theta = theta
        
        # Local refinement
        if min_side_for_this_cluster < best_side:
            refined_theta, refined_side = _refine_theta(centers, hex_verts, best_theta, n_steps=100)
            if refined_side < min_side_for_this_cluster:
                min_side_for_this_cluster = refined_side
                best_theta = refined_theta
        
        if min_side_for_this_cluster < best_side:
            best_side = min_side_for_this_cluster
            inner_hex_data = np.column_stack([centers, np.zeros(11)])
            outer_hex_data = np.array([0, 0, np.rad2deg(best_theta)])
            best_config = (inner_hex_data, outer_hex_data, best_side)
    
    if best_config is None:
        return _fallback()

    return best_config[0], best_config[1], best_config[2]

def _fallback():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
