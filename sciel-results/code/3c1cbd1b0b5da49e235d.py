"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Arrange 12 unit hexagons in a hexagonal lattice pattern
    # Rows: 2, 3, 4, 3 hexagons
    # Flat-top orientation: width = sqrt(3), height = 2
    # Horizontal center spacing: sqrt(3)
    # Vertical row spacing: 1.5
    # Alternating rows offset by sqrt(3)/2
    
    s = 1.0  # side length of inner hexagons
    hw = np.sqrt(3) * s / 2  # half-width
    hh = s  # half-height (point to center)
    
    inner = []
    rows = [2, 3, 4, 3]
    row_y = [0, -1.5, -3.0, -4.5]
    
    for i, count in enumerate(rows):
        y = row_y[i]
        x_start = -(count - 1) * np.sqrt(3) * s / 2
        for j in range(count):
            x = x_start + j * np.sqrt(3) * s
            inner.append([x, y, 0.0])
    
    # Compute bounding box of all hexagon vertices
    all_vertices = []
    for x, y, angle in inner:
        # Vertices of unit hexagon at (x, y) with flat-top orientation
        for k in range(6):
            theta = np.radians(60 * k)
            vx = x + s * np.cos(theta)
            vy = y + s * np.sin(theta)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # The outer hexagon must contain all these vertices
    # For a regular hexagon with center (cx, cy), side L, and rotation phi
    # A point (px, py) is inside if max over 6 directions of projection <= L*sqrt(3)/2
    # (the apothem is L*sqrt(3)/2)
    
    # Try center at centroid of vertices
    cx = np.mean(all_vertices[:, 0])
    cy = np.mean(all_vertices[:, 1])
    
    # For a given rotation, find minimum L such that all points are inside
    # The constraint: for each of 6 directions, max projection <= L * sqrt(3)/2
    # So L >= max_over_points, max_over_dirs (dot product) / (sqrt(3)/2)
    
    best_L = float('inf')
    best_phi = 0
    
    # Try several rotations
    for phi_deg in range(0, 60, 5):
        phi = np.radians(phi_deg)
        # 6 normal directions for the hexagon
        dirs = []
        for k in range(6):
            angle = phi + np.radians(60 * k)
            dirs.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for v in all_vertices:
            rel = v - np.array([cx, cy])
            for d in dirs:
                proj = np.dot(rel, d)
                if proj > max_proj:
                    max_proj = proj
        
        L = max_proj / (np.sqrt(3) / 2)
        if L < best_L:
            best_L = L
            best_phi = phi_deg
    
    # Refine with finer search
    best_L = float('inf')
    best_phi = 0
    for phi_deg in np.arange(0, 60, 0.5):
        phi = np.radians(phi_deg)
        dirs = []
        for k in range(6):
            angle = phi + np.radians(60 * k)
            dirs.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0
        for v in all_vertices:
            rel = v - np.array([cx, cy])
            for d in dirs:
                proj = np.dot(rel, d)
                if proj > max_proj:
                    max_proj = proj
        
        L = max_proj / (np.sqrt(3) / 2)
        if L < best_L:
            best_L = L
            best_phi = phi_deg
    
    return inner, [cx, cy], best_L, best_phi
# EVOLVE-BLOCK-END
