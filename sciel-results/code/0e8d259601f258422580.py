import numpy as np
from scipy.optimize import minimize
import time

def get_hex_vertices(cx, cy, s=1.0, angle_deg=0):
    """
    Returns vertices of a regular hexagon.
    """
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.arange(0, 360, 60))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def min_hex_side_length(points):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, angle 0)
    that contains all given points.
    """
    if len(points) == 0:
        return 0.0
    
    # Normals for hexagon with angle 0
    # Edges are perpendicular to vectors at 0, 60, 120, 180, 240, 300 degrees
    normals = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    # Project points onto normals
    projections = points @ normals.T
    # We need max absolute projection
    max_proj = np.max(np.abs(projections))
    
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def objective(params):
    """
    Objective function for optimization.
    params: array of shape (22,) representing 11 (x,y) pairs.
    """
    pts = params.reshape(11, 2)
    
    # Re-center to prevent drift
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    # Get all vertices of all hexagons
    all_vertices = []
    for i in range(11):
        verts = get_hex_vertices(pts[i, 0], pts[i, 1])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Calculate required side length
    side_len = min_hex_side_length(all_vertices)
    
    # Calculate overlap penalty
    penalty = 0.0
    min_dist = np.sqrt(3.0)
    for i in range(11):
        for j in range(i + 1, 11):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
                
    # Scale penalty to be significant
    return side_len + 100.0 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 630  # Safety margin for 600s limit
    
    # Initial configuration: Symmetric 11-hexagon
    # Center + 6 neighbors + 4 more
    # Lattice vectors
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Indices for a compact 11-hex shape
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 6 neighbors
    # (2,-1), (-2,1), (1,1), (-1,-1) -> 4 more?
    # Let's try a different set for better symmetry.
    # How about a "plus" sign of width 3? No, that's 5.
    # Let's just use the 7 central ones and add 4 at the "corners" of the 2nd ring.
    # Corners of 2nd ring relative to center:
    # (2,0), (-2,0), (1,2), (-1,2), (1,-2), (-1,-2) ?
    # Let's stick to a simple random restart strategy around the 7-hex core.
    
    best_side = np.inf
    best_params = None
    
    # Generate a few initial guesses
    inits = []
    
    # Init 1: 7-hex core + 4 hexes pushed out randomly
    base_pts = np.zeros((11, 2))
    # 7-hex core
    base_pts[0] = [0, 0]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        base_pts[i+1] = np.array([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    # Add 4 more, place them at distance 2*sqrt(3) from center at 45, 135, 225, 315 deg
    for i in range(4):
        angle = np.deg2rad(45 + 90 * i)
        base_pts[7+i] = np.array([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
    inits.append(base_pts.flatten())
    
    # Init 2: Random jittered version of Init 1
    inits.append(base_pts.flatten() + np.random.normal(0, 0.5, 22))
    
    # Init 3: A different symmetric configuration
    # Try a line? No, hexagon is better.
    # Try 3-3-3-2?
    # Let's just run optimization on the first init.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        res = minimize(objective, init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success or res.fun < best_side:
            # Check if valid (low penalty)
            params = res.x
            pts = params.reshape(11, 2)
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            
            # Verify no overlap
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if np.linalg.norm(pts[i] - pts[j]) < np.sqrt(3.0) - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                all_vertices = []
                for i in range(11):
                    verts = get_hex_vertices(pts[i, 0], pts[i, 1])
                    all_vertices.append(verts)
                all_vertices = np.vstack(all_vertices)
                side_len = min_hex_side_length(all_vertices)
                
                if side_len < best_side:
                    best_side = side_len
                    best_params = params
                    best_pts = pts
        else:
            # Even if not success, if fun is lower, check validity
            if res.fun < best_side:
                 params = res.x
                 pts = params.reshape(11, 2)
                 centroid = np.mean(pts, axis=0)
                 pts = pts - centroid
                 valid = True
                 for i in range(11):
                     for j in range(i+1, 11):
                         if np.linalg.norm(pts[i] - pts[j]) < np.sqrt(3.0) - 1e-6:
                             valid = False
                             break
                     if not valid:
                         break
                 if valid:
                     all_vertices = []
                     for i in range(11):
                         verts = get_hex_vertices(pts[i, 0], pts[i, 1])
                         all_vertices.append(verts)
                     all_vertices = np.vstack(all_vertices)
                     side_len = min_hex_side_length(all_vertices)
                     if side_len < best_side:
                         best_side = side_len
                         best_params = params
                         best_pts = pts

    # If optimization failed or didn't find a good one, fallback to a known valid packing
    # The seed program's packing is valid but large (side 8).
    # We should ensure we return at least the seed's quality if possible.
    # However, our optimizer should easily beat side 8.
    
    if best_params is None:
        # Fallback to seed-like configuration but centered
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

    # Construct return values
    # Inner hexagons: (x, y, 0)
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = best_pts[:, 0]
    inner_hex_data[:, 1] = best_pts[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    # Outer hexagon: centered at 0,0, angle 0
    outer_hex_data = np.array([0, 0, 0])
    
    # Side length
    # Ensure it's slightly larger than calculated to account for floating point errors
    outer_hex_side_length = best_side * (1 + 1e-6)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
