<<<<<<< SEARCH
"""Improved circle packing using trust-constr NLP with explicit constraints."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import itertools

def _initial_guess():
    """Generate an initial guess with circles in corners and center."""
    # Place circles in a pattern that uses corners effectively
    # Corners: 4, edges: 8, interior: 14
    positions = []
    radii = []
    
    # 4 corner circles
    corner_r = 0.18
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for x, y in corners:
        positions.append((x, y))
        radii.append(corner_r)
    
    # 8 edge circles (2 per edge, avoiding corners)
    edge_r = 0.12
    # Top edge
    positions.extend([(0.3, 1-edge_r), (0.7, 1-edge_r)])
    radii.extend([edge_r, edge_r])
    # Bottom edge
    positions.extend([(0.3, edge_r), (0.7, edge_r)])
    radii.extend([edge_r, edge_r])
    # Left edge
    positions.extend([(edge_r, 0.3), (edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    # Right edge
    positions.extend([(1-edge_r, 0.3), (1-edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    
    # 14 interior circles in a grid-like pattern
    interior_r = 0.08
    # 3x3 grid in the center region, plus 1 extra
    cx, cy = 0.5, 0.5
    spacing = 0.22
    for i in range(-1, 2):
        for j in range(-1, 2):
            if len(positions) < N:
                positions.append((cx + i*spacing, cy + j*spacing))
                radii.append(interior_r)
    
    # Add remaining circles if needed
    while len(positions) < N:
        positions.append((0.5, 0.5))
        radii.append(0.05)
    
    # Flatten to array: [x0, y0, r0, x1, y1, r1, ...]
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    
    return np.concatenate([x, y, r])
=======
"""Improved circle packing using trust-constr NLP with explicit constraints."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import itertools

def _voronoi_radii(points, margin=0.0):
    """Compute max radii for each circle based on nearest neighbor distances and boundaries."""
    n = len(points)
    pts = np.array(points)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        dist_boundary = min(pts[i][0], 1-pts[i][0], pts[i][1], 1-pts[i][1])
        # Distance to other circles
        min_dist = dist_boundary
        for j in range(n):
            if i != j:
                d = np.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
                min_dist = min(min_dist, d/2)
        radii[i] = min_dist - margin
    return np.maximum(radii, 0.01)

def _initial_guess():
    """Generate an initial guess with circles in corners, edges, and interior."""
    positions = []
    
    # 4 corner circles
    corner_r = 0.15
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for x, y in corners:
        positions.append((x, y))
    
    # 8 edge circles (2 per edge, avoiding corners)
    edge_r = 0.10
    # Top edge
    positions.extend([(0.3, 1-edge_r), (0.7, 1-edge_r)])
    # Bottom edge
    positions.extend([(0.3, edge_r), (0.7, edge_r)])
    # Left edge
    positions.extend([(edge_r, 0.3), (edge_r, 0.7)])
    # Right edge
    positions.extend([(1-edge_r, 0.3), (1-edge_r, 0.7)])
    
    # 14 interior circles in a hexagonal-like pattern
    cx, cy = 0.5, 0.5
    # 3x3 grid
    for i in range(-1, 2):
        for j in range(-1, 2):
            if len(positions) < N:
                positions.append((cx + i*0.2, cy + j*0.2))
    
    # Add remaining circles if needed
    while len(positions) < N:
        positions.append((0.5 + 0.1*np.random.randn(), 0.5 + 0.1*np.random.randn()))
    
    positions = positions[:N]
    
    # Compute Voronoi-based radii
    radii = _voronoi_radii(positions, margin=0.005)
    
    # Flatten to array: [x0, y0, r0, x1, y1, r1, ...]
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    r = radii
    
    return np.concatenate([x, y, r])
>>>>>>> REPLACE

