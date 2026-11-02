import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Hexagonal lattice
    # Generate lattice points
    # Hex lattice basis:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    # Distance between adjacent centers is sqrt(3).
    # Wait, side length is 1.
    # Distance between centers of touching hexagons is sqrt(3).
    # Let's verify.
    # Hexagon vertices: (1,0), (0.5, sqrt(3)/2), ...
    # Edge midpoint: (0.75, sqrt(3)/4)? No.
    # Midpoint of edge between (1,0) and (0.5, sqrt(3)/2) is (0.75, sqrt(3)/4).
    # Distance from center (0,0) to this midpoint is sqrt(0.75^2 + (sqrt(3)/4)^2) = sqrt(9/16 + 3/16) = sqrt(12/16) = sqrt(3)/2.
    # So inradius r = sqrt(3)/2.
    # Distance between centers of two touching hexagons is 2r = sqrt(3).
    
    # Lattice vectors
    a = np.sqrt(3)
    b = 1.5
    v1 = np.array([a, 0])
    v2 = np.array([a/2, b])
    
    # Generate points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # Select 11 closest to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:N]
    x0 = points[indices].flatten()
    
    # Objective function
    def objective(x):
        if time.time() > deadline:
            return 1e6
        pos = x.reshape(N, 2)
        
        # Calculate required S
        # Vertices of unit hexagon (flat top? No, standard orientation)
        # Let's use standard orientation: vertices at 0, 60, ... degrees
        angles = np.deg2rad(np.arange(0, 360, 60))
        verts_local = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = 0.0
        for p in pos:
            verts_global = p + verts_local
            # Project onto 6 normals
            # Normals for flat-top hexagon?
            # If vertices are at 0, 60..., the normals are at 30, 90, ...
            # Let's check.
            # Edge from 0 to 60 deg. Midpoint angle 30 deg.
            # Normal is at 30 deg.
            # So normals are at 30, 90, 150, 210, 270, 330.
            n_angles = np.deg2rad(np.arange(30, 360, 60))
            normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
            # Projection of each vertex onto each normal
            # proj = verts_global @ normals.T
            # We need max over all vertices and normals
            proj = verts_global @ normals.T
            current_max = np.max(proj)
            if current_max > max_proj:
                max_proj = current_max
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d) ** 2
        
        return S + 10.0 * penalty

    # Optimization
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if time.time() > deadline:
        # Fallback to initial guess if timeout
        best_x = x0
    else:
        best_x = res.x
    
    pos = best_x.reshape(N, 2)
    
    # Calculate final S
    angles = np.deg2rad(np.arange(0, 360, 60))
    verts_local = np.column_stack([np.cos(angles), np.sin(angles)])
    n_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    max_proj = 0.0
    for p in pos:
        verts_global = p + verts_local
        proj = verts_global @ normals.T
        current_max = np.max(proj)
        if current_max > max_proj:
            max_proj = current_max
    
    S = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Format output
    inner = []
    for p in pos:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0

