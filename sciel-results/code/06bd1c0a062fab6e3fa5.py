import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # 1. Define local coordinates for 12 hexagons in a 3x4 grid
    # Axial coordinates (q, r)
    # q: 0..3, r: 0..2
    points = []
    for r in range(3):
        for q in range(4):
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            points.append([x, y])
    points = np.array(points)
    
    # Center the cluster
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    # 2. Precompute normals for the outer hexagon (fixed orientation 0)
    # Normals are perpendicular to sides.
    # Sides are at angles 0, 60, ... if vertices are at 30, 90...
    # Let's use standard orientation: vertices at 0, 60, 120...
    # Then sides are at 30, 90, 150...
    # Normals are at 30, 90, 150, 210, 270, 330.
    angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_S = float('inf')
    best_psi = 0.0
    best_tx = 0.0
    best_ty = 0.0
    
    # 3. Search over rotation psi
    # Step 0.5 degrees
    for psi_deg in np.arange(0, 60, 0.5):
        if time.time() > deadline:
            break
            
        psi = np.deg2rad(psi_deg)
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate points
        p_rot = points @ R.T
        
        # Compute midpoints for each normal direction
        # Projections: p_rot @ normals.T
        proj = p_rot @ normals.T  # Shape (12, 6)
        mins = np.min(proj, axis=0)
        maxs = np.max(proj, axis=0)
        mids = (mins + maxs) / 2.0
        
        # Solve for optimal translation t
        # We want normals @ t = -mids
        # A t = b
        A = normals
        b = -mids
        # Least squares solution
        try:
            t_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            tx_ls, ty_ls = t_ls
        except:
            tx_ls, ty_ls = 0.0, 0.0
            
        # Local search around LSQ solution
        # Grid of 5x5 around (tx_ls, ty_ls) with step 0.1
        best_local_S = float('inf')
        best_local_tx, best_local_ty = tx_ls, ty_ls
        
        for dx in np.arange(-0.2, 0.21, 0.1):
            for dy in np.arange(-0.2, 0.21, 0.1):
                tx = tx_ls + dx
                ty = ty_ls + dy
                t = np.array([tx, ty])
                
                # Translate points
                p_trans = p_rot + t
                
                # Compute max absolute projection
                proj_trans = p_trans @ normals.T
                max_abs = np.max(np.abs(proj_trans))
                
                S = max_abs * (2.0 / np.sqrt(3))
                
                if S < best_local_S:
                    best_local_S = S
                    best_local_tx, best_local_ty = tx, ty
                    
        if best_local_S < best_S:
            best_S = best_local_S
            best_psi = psi
            best_tx = best_local_tx
            best_ty = best_local_ty
            
    # 4. Construct final output
    c, s = np.cos(best_psi), np.sin(best_psi)
    R = np.array([[c, -s], [s, c]])
    final_points = points @ R.T
    final_points += np.array([best_tx, best_ty])
    
    inner = [[float(p[0]), float(p[1]), float(np.rad2deg(best_psi))] for p in final_points]
    center = (0.0, 0.0)
    side = float(best_S)
    angle = 0.0
    
    return inner, center, side, angle
