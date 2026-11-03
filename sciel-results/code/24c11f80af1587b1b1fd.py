"""
Strategy: Discrete Lattice Search with Convex Hull Optimization
"""

import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

N = 12

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _axial_to_pixel(q, r):
    x = 1.5 * q
    y = np.sqrt(3) * (r + q / 2.0)
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 1. Generate candidate sets of 12 lattice points
    # Base points: 19 points in a hexagon of radius 2
    base_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                base_points.append((q, r))
                
    base_pixels = [_axial_to_pixel(q, r) for q, r in base_points]
    
    best_side = float('inf')
    best_config = None
    best_outer_angle = 0
    best_outer_center = (0.0, 0.0)
    
    # Pre-compute the 6 vertex offsets for a hexagon at (0,0) angle 0
    hex_offsets = []
    for k in range(6):
        a = k * np.pi / 3
        hex_offsets.append((np.cos(a), np.sin(a)))
    hex_offsets = np.array(hex_offsets)
    
    count = 0
    for subset_indices in itertools.combinations(range(len(base_points)), 12):
        if time.time() > deadline - 5:
            break
            
        # Get pixel coords
        pts = np.array([base_pixels[i] for i in subset_indices])
        
        # Center the cluster
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        
        # Generate all 72 vertices
        # Shape: (72, 2)
        # centered_pts[:, np.newaxis, :] + hex_offsets[np.newaxis, :, :]
        all_verts = centered_pts[:, np.newaxis, :] + hex_offsets[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 2)
        
        # Compute Convex Hull
        try:
            hull = ConvexHull(all_verts)
            hull_verts = all_verts[hull.vertices]
        except Exception:
            # If all points are collinear or something weird, use all points
            hull_verts = all_verts
            
        # Check 6 outer angles
        # For each outer angle, we need to project hull_verts onto 6 normal directions
        # and find the max projection.
        
        # Directions for outer angle oa:
        # n_k = (cos(oa + k*pi/3 + pi/6), sin(oa + k*pi/3 + pi/6))
        
        # We can pre-compute the 36 normal vectors (6 angles * 6 directions)
        # But oa changes.
        
        for oa in [0, 30, 60, 90, 120, 150]:
            max_rad = 0
            oa_rad = np.radians(oa)
            for k in range(6):
                theta = oa_rad + k * np.pi / 3 + np.pi / 6
                nx = np.cos(theta)
                ny = np.sin(theta)
                # Project hull vertices
                projections = hull_verts[:, 0] * nx + hull_verts[:, 1] * ny
                max_proj = np.max(projections)
                if max_proj > max_rad:
                    max_rad = max_proj
            
            side = 2.0 * max_rad / np.sqrt(3)
            
            if side < best_side:
                best_side = side
                best_config = [(float(x), float(y), 0.0) for x, y in centered_pts]
                best_outer_angle = oa
                best_outer_center = (0.0, 0.0)
        
        count += 1

    # Fallback
    if best_config is None:
        positions = []
        for i in range(4):
            for j in range(3):
                q = i - 1.5
                r = j - 1
                x, y = _axial_to_pixel(q, r)
                positions.append((x, y, 0.0))
        best_side = 10.0
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0

    return best_config, best_outer_center, float(best_side), float(best_outer_angle)

# EVOLVE-BLOCK-END
