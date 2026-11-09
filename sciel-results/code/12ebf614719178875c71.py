import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Initial guess: 3x4 staggered grid
    d = np.sqrt(3)
    # y coordinates for 3 rows
    ys = [-1.5, 0.0, 1.5]
    # x coordinates for each row (4 hexagons each)
    # Row 0 and 2 are aligned, Row 1 is offset by d/2
    xs_row_even = [-1.5*d, -0.5*d, 0.5*d, 1.5*d]
    xs_row_odd = [-1.0*d, 0.0*d, 1.0*d, 2.0*d]
    
    points = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            for x in xs_row_even:
                points.append([x, y])
        else:
            for x in xs_row_odd:
                points.append([x, y])
                
    points = np.array(points)
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Variables: x0, y0, ..., x11, y11, alpha
    # Total 25 variables
    x0 = points.flatten()
    alpha0 = 0.0
    v0 = np.concatenate([x0, [alpha0]])
    
    # Normals for outer hexagon (angle 0)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles_outer = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    def objective(v):
        pts = v[:2*N].reshape(N, 2)
        alpha = v[2*N]
        
        # Calculate vertices of all inner hexagons
        # Inner hexagon vertices relative to center: (cos(k*pi/3 + alpha), sin(k*pi/3 + alpha))
        k = np.arange(6) * np.pi / 3
        rel_vertices = np.column_stack([np.cos(k + alpha), np.sin(k + alpha)])
        
        # All vertices in absolute coords
        # pts: (N, 2), rel_vertices: (6, 2)
        # We need to compute max dot product with normals
        # For each point i, and each vertex k, and each normal m:
        # dot = (pts[i] + rel_vertices[k]) dot normals[m]
        
        # Vectorized:
        # Add rel_vertices to each point
        # shape (N, 1, 2) + (1, 6, 2) -> (N, 6, 2)
        all_vertices = pts[:, np.newaxis, :] + rel_vertices[np.newaxis, :, :]
        
        # Dot product with normals (6, 2)
        # all_vertices: (N, 6, 2)
        # normals: (6, 2)
        # We want max over N, 6 (vertices), 6 (normals)
        # proj = np.tensordot(all_vertices, normals, axes=([2], [1]))
        # shape (N, 6, 6)
        proj = np.einsum('ijk,kl->ijl', all_vertices, normals)
        
        max_proj = np.max(proj)
        
        # Minimal L is (2/sqrt(3)) * max_proj
        L_val = (2.0 / np.sqrt(3.0)) * max_proj
        
        # Penalty for overlap
        # Distance between all pairs
        # pts: (N, 2)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        min_dist = np.min(dists[mask])
        
        penalty = 0.0
        if min_dist < np.sqrt(3.0):
            # Strong penalty
            penalty = 1000.0 * (np.sqrt(3.0) - min_dist) ** 2
            
        return L_val + penalty

    # Optimize
    res = minimize(objective, v0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    v_opt = res.x
    pts_opt = v_opt[:2*N].reshape(N, 2)
    alpha_opt = v_opt[2*N]
    
    # Calculate final L
    k = np.arange(6) * np.pi / 3
    rel_vertices = np.column_stack([np.cos(k + alpha_opt), np.sin(k + alpha_opt)])
    all_vertices = pts_opt[:, np.newaxis, :] + rel_vertices[np.newaxis, :, :]
    proj = np.einsum('ijk,kl->ijl', all_vertices, normals)
    max_proj = np.max(proj)
    L_final = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Format output
    inner = []
    for i in range(N):
        x, y = pts_opt[i]
        angle_deg = np.rad2deg(alpha_opt)
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), L_final, 0.0

# EVOLVE-BLOCK-END
