"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 1. Generate initial hex grid positions
    # Axial coordinates to Cartesian (flat-top hexagons, but we use pointy-top vertices in _hex_vertices? 
    # Let's stick to the convention: vertices at 30, 90... means pointy top? 
    # _hex_vertices: a + pi/6 + k*pi/3. If a=0, angles are 30, 90... 
    # Vertex at 90 is (0, 1). Vertex at 30 is (sqrt(3)/2, 0.5).
    # This is a "pointy-top" hexagon.
    # Neighbors in a pointy-top hex grid:
    # East: (sqrt(3), 0)
    # NE: (sqrt(3)/2, 1.5)
    # NW: (-sqrt(3)/2, 1.5)
    # West: (-sqrt(3), 0)
    # SW: (-sqrt(3)/2, -1.5)
    # SE: (sqrt(3)/2, -1.5)
    # Distance between centers: sqrt(3) for E/W, sqrt(3) for others?
    # dist((0,0), (sqrt(3),0)) = sqrt(3).
    # dist((0,0), (sqrt(3)/2, 1.5)) = sqrt(3/4 + 2.25) = sqrt(3).
    # Yes, all neighbors are at distance sqrt(3).
    
    def hex_grid_positions(n):
        positions = [(0.0, 0.0)]
        # Directions for pointy-top hex grid
        # (dx, dy)
        dirs = [
            (sqrt3, 0),
            (sqrt3/2, 1.5),
            (-sqrt3/2, 1.5),
            (-sqrt3, 0),
            (-sqrt3/2, -1.5),
            (sqrt3/2, -1.5)
        ]
        
        # BFS to get compact cluster
        # Queue of (x, y, q, r) or just (x,y) and check if we've seen it?
        # Better to use axial coords to avoid floating point errors in grid generation
        axial = [(0,0)]
        axial_dirs = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        
        # We need to map axial to cartesian for pointy-top
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        # Let's verify neighbor distances
        # (0,0) -> (0,0)
        # (1,0) -> (sqrt(3), 0). Dist sqrt(3). Correct.
        # (0,1) -> (sqrt(3)/2, 1.5). Dist sqrt(3). Correct.
        
        q_r = [(0,0)]
        visited = set([(0,0)])
        
        while len(positions) < n:
            next_q_r = []
            for q, r in q_r:
                for dq, dr in axial_dirs:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in visited:
                        visited.add((nq, nr))
                        next_q_r.append((nq, nr))
            for nq, nr in next_q_r:
                if len(positions) < n:
                    x = sqrt3 * (nq + nr / 2.0)
                    y = 1.5 * nr
                    positions.append((x, y))
            q_r = next_q_r
        return np.array(positions)

    x0 = hex_grid_positions(N)
    
    # Objective function
    # x is a flat array of 2*N coordinates
    # We want to minimize the max width in the 6 hexagonal directions
    # Plus a penalty for overlapping (dist < sqrt(3))
    
    # Normals for the 6 sides of the outer hexagon (pointy-top? or flat-top?)
    # The outer hexagon in the current code uses _hex_vertices with angle 0.
    # _hex_vertices(0,0,1,0) gives vertices at 30, 90, 150, 210, 270, 330.
    # This is a pointy-top hexagon.
    # The sides are perpendicular to the directions:
    # Side between 30 and 90: Bisector at 60 deg. Normal is 60 deg?
    # No, the normal to the side is the direction of the bisector of the angle between the two vertices?
    # Vertices at 30 and 90. Midpoint angle 60.
    # The side is perpendicular to the radius at 60 deg?
    # Let's check the constraint in _point_in_hex:
    # theta_k = (2k+1)*pi/6.
    # k=0: 30 deg.
    # k=1: 90 deg.
    # k=2: 150 deg.
    # k=3: 210 deg.
    # k=4: 270 deg.
    # k=5: 330 deg.
    # These are the normals to the 6 sides.
    
    thetas = np.array([(2*k+1)*np.pi/6 for k in range(6)])
    normals = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    def objective(x):
        pos = x.reshape(-1, 2)
        
        # Calculate widths in each of the 6 normal directions
        # proj = pos @ normals.T
        proj = pos @ normals.T
        widths = np.max(proj, axis=0) - np.min(proj, axis=0)
        
        # Required side length S
        # S * sqrt(3) >= max(width)
        S = np.max(widths) / sqrt3
        
        # Overlap penalty
        # dist matrix
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Penalty for dist < sqrt(3)
        # We use a quadratic penalty for violations
        violations = np.maximum(0, sqrt3 - dists)
        # Zero out diagonal
        np.fill_diagonal(violations, 0)
        penalty = np.sum(violations**2)
        
        return S + 100.0 * penalty

    # Initial guess
    x_flat = x0.flatten()
    
    # Optimize
    # Use Powell method which doesn't require gradients and handles non-smoothness (max) well
    res = minimize(objective, x_flat, method='Powell', options={'maxiter': 2000, 'disp': False})
    
    if time.time() > deadline - 5.0:
        # If we ran out of time, use the best found so far or initial
        # Powell returns res.x
        pass
        
    x_opt = res.x
    pos_opt = x_opt.reshape(-1, 2)
    
    # Re-center the cluster to (0,0) to minimize the outer hexagon size if it drifted?
    # The objective function is translation invariant (widths don't change).
    # So the optimizer might drift. We should re-center the result to the centroid.
    centroid = np.mean(pos_opt, axis=0)
    pos_opt = pos_opt - centroid
    
    # Calculate final side length
    proj = pos_opt @ normals.T
    widths = np.max(proj, axis=0) - np.min(proj, axis=0)
    best_side = np.max(widths) / sqrt3
    
    # Verify validity (just in case)
    # Check distances
    diff = pos_opt[:, np.newaxis, :] - pos_opt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 100.0)
    if np.min(dists) < sqrt3 - 1e-9:
        # If overlap, fall back to initial grid
        pos_opt = x0 - np.mean(x0, axis=0)
        proj = pos_opt @ normals.T
        widths = np.max(proj, axis=0) - np.min(proj, axis=0)
        best_side = np.max(widths) / sqrt3

    # Add tiny epsilon
    best_side *= (1.0 + 1e-6)
    
    inner = [[pos_opt[i][0], pos_opt[i][1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
