# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin from 150s

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _point_in_polygon(pt, verts):
    """Check if point is inside polygon using winding number / cross products. Assumes CCW."""
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < 0:
            return False
    return True

def construct():
    # N = 12
    N = 12
    
    # Define lattice vectors for hexagonal packing
    # Scale factor 2/sqrt(3) ensures center distance >= 2 for unit hexagons
    scale = 2.0 / np.sqrt(3)
    v1 = np.array([1.5 * scale, (np.sqrt(3)/2) * scale]) # [sqrt(3), 1]
    v2 = np.array([0, np.sqrt(3) * scale])               # [0, 2]
    
    # Generate 3x4 grid of centers
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * v1 + j * v2
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the cluster at origin
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    
    # Precompute inner vertices for all centers (angle 0)
    inner_verts_list = []
    for c in centers:
        inner_verts_list.append(_hex_verts(c[0], c[1], 1.0, 0))
        
    def is_valid_packing(S, outer_angle):
        outer_verts = _hex_verts(0, 0, S, outer_angle)
        for inner_verts in inner_verts_list:
            for v in inner_verts:
                if not _point_in_polygon(v, outer_verts):
                    return False
        return True
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Try several outer angles
    for angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        low = 2.0
        high = 10.0
        
        # Ensure high is valid
        while not is_valid_packing(high, angle):
            high *= 1.5
            if time.time() - _START_TIME > _TIME_LIMIT:
                break
                
        # Binary search
        for _ in range(50):
            if time.time() - _START_TIME > _TIME_LIMIT:
                break
            mid = (low + high) / 2
            if is_valid_packing(mid, angle):
                high = mid
            else:
                low = mid
                
        if high < best_S:
            best_S = high
            best_angle = angle
            
    # Prepare result
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_S), float(best_angle)

# EVOLVE-BLOCK-END
