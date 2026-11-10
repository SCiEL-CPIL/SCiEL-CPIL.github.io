import numpy as np
import time

def hexagon_packing_12():
    # Start time for safety
    start_time = time.time()
    
    # Lattice vectors for hexagons of side 1
    # Distance between centers of touching hexes is sqrt(3)
    # To ensure separation, we scale the lattice spacing by 1.001
    scale = 1.001
    d = np.sqrt(3) * scale
    
    e1 = np.array([d, 0])
    e2 = np.array([d/2, d * np.sqrt(3)/2])
    
    # Define 12 lattice coordinates (q, r)
    # Configuration: 
    # r=0: q=-2, -1, 0, 1, 2 (5)
    # r=1: q=-1, 0, 1 (3)
    # r=-1: q=-1, 0, 1 (3)
    # r=2: q=0 (1)
    # Total 12
    coords = [
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2)
    ]
    
    centers = np.array([q * e1 + r * e2 for q, r in coords])
    
    # Center the cluster at origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Inner hexagon angle
    # Knowledge says 30 degrees
    angle_deg = 30.0
    inner_angles = np.full(12, angle_deg)
    
    # Get all vertices of all inner hexagons
    def get_vertices(cx, cy, ang_deg):
        ang = np.radians(ang_deg)
        verts = []
        for k in range(6):
            theta = ang + k * np.pi / 3
            verts.append([cx + np.cos(theta), cy + np.sin(theta)])
        return np.array(verts)
    
    all_verts = np.vstack([get_vertices(cx, cy, angle_deg) for cx, cy in centers])
    
    # Find optimal outer hexagon
    # Minimize S = max_k (max_proj_k - min_proj_k) / sqrt(3)
    best_s = np.inf
    best_phi = 0
    best_center = np.zeros(2)
    
    # Scan angles
    for phi_deg in np.linspace(0, 60, 601):
        phi = np.radians(phi_deg)
        normals = np.array([[np.cos(phi + k * np.pi/3), np.sin(phi + k * np.pi/3)] for k in range(6)])
        
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        spans = max_projs - min_projs
        s_candidate = np.max(spans) / np.sqrt(3)
        
        if s_candidate < best_s:
            best_s = s_candidate
            best_phi = phi
            # Calculate optimal center
            # Center projection on n_k should be midpoint of span
            # c . n_k = (max_k + min_k) / 2
            # We have 6 equations, 2 unknowns. Use least squares on first 3 (independent)
            A = normals[:3]
            b = (max_projs[:3] + min_projs[:3]) / 2
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            best_center = c
    
    # Construct output
    inner = np.column_stack([centers, inner_angles])
    outer = np.array([best_center[0], best_center[1], np.degrees(best_phi)])
    s = best_s
    
    return inner, outer, s
