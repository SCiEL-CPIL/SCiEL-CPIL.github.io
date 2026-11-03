import math
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def compute_min_side(points, angle_deg):
    # points: Nx2 array
    # angle_deg: rotation of outer hexagon
    a = math.radians(angle_deg)
    # Normals to the sides of a regular hexagon
    # Sides are at angles a+30, a+90, a+150 (normals)
    # Actually, if vertices are at a, a+60..., sides are perpendicular to normals at a+30, a+90...
    normals = []
    for k in range(3):
        theta = a + math.pi/6 + k * math.pi/3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_r = 0
    for n in normals:
        proj = points @ n
        min_p = np.min(proj)
        max_p = np.max(proj)
        r = (max_p - min_p) / 2.0
        if r > max_r:
            max_r = r
            
    # S = 2 * r_in / sqrt(3)
    side = 2.0 * max_r / math.sqrt(3)
    return side

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin for 120s limit
    
    # Define lattice vectors for pointy-top hexagons
    # u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    # Scale by 1.01 to ensure disjointness (gap between hexagons)
    scale = 1.01
    u = np.array([math.sqrt(3) * scale, 0.0])
    v = np.array([math.sqrt(3)/2 * scale, 1.5 * scale])
    
    # 12 indices (i, j)
    # Center + 6 neighbors + 5 more
    indices = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2), (1,1)
    ]
    
    centers = []
    for i, j in indices:
        c = i * u + j * v
        centers.append(c)
        
    # Inner hexagon angle: 90 degrees (pointy-top)
    inner_angle = 90.0
    
    # Generate all vertices of the 12 hexagons
    all_verts = []
    for c in centers:
        verts = hex_vertices(c[0], c[1], inner_angle, 1.0)
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Optimize outer hexagon angle
    # Scan 0 to 60 degrees
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse scan
    # Check time
    if time.time() > deadline:
        # Return a valid fallback if we ran out of time (unlikely in coarse scan)
        # But we need to return something valid.
        # Let's just do the best we have so far. If no scan done, use default.
        pass

    for a in np.arange(0, 60, 0.5):
        if time.time() > deadline:
            break
        side = compute_min_side(all_verts, a)
        if side < best_side:
            best_side = side
            best_angle = a
            
    if time.time() > deadline:
        # Calculate center and return
        a = math.radians(best_angle)
        normals = []
        for k in range(3):
            theta = a + math.pi/6 + k * math.pi/3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        mids = []
        for n in normals:
            proj = all_verts @ n
            mids.append((np.min(proj) + np.max(proj)) / 2.0)
        A = np.array(normals)
        b = np.array(mids)
        C, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        best_center = (C[0], C[1])
        
        inner = []
        for c in centers:
            inner.append([c[0], c[1], inner_angle])
        return inner, best_center, best_side, best_angle

    # Fine scan around best
    for a in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
        if time.time() > deadline:
            break
        side = compute_min_side(all_verts, a)
        if side < best_side:
            best_side = side
            best_angle = a
            
    # Calculate center for best_angle
    a = math.radians(best_angle)
    normals = []
    for k in range(3):
        theta = a + math.pi/6 + k * math.pi/3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
    mids = []
    for n in normals:
        proj = all_verts @ n
        mids.append((np.min(proj) + np.max(proj)) / 2.0)
        
    A = np.array(normals) # 3x2
    b = np.array(mids)    # 3
    
    C, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    best_center = (C[0], C[1])
    
    # Construct return values
    inner = []
    for c in centers:
        inner.append([c[0], c[1], inner_angle])
        
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
