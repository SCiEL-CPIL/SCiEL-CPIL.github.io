import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Unit hexagon vertices (pointy top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for the container hexagon (pointy top, same orientation)
    # The sides are perpendicular to the directions 30, 90, 150, 210, 270, 330
    # Wait, if vertices are at 0, 60..., sides are centered at 30, 90...
    # Normal vectors for the 6 sides of the container:
    c_angles = np.deg2rad(np.arange(30, 390, 60))
    normals = np.column_stack([np.cos(c_angles), np.sin(c_angles)])
    
    def calc_S(positions):
        # positions: (N, 2)
        # Calculate max projection of all vertices onto all normals
        # For each hexagon i, vertices are positions[i] + hex_verts
        # We need max over all i, v, k of (positions[i] + hex_verts[v]) dot normals[k]
        
        # Vectorized:
        # pos: (N, 1, 2)
        # verts: (1, 6, 2)
        # dots: (N, 6, 6) -> N hexagons, 6 verts, 6 normals
        pos = positions[:, np.newaxis, :] # (N, 1, 2)
        verts = hex_verts[np.newaxis, :, :] # (1, 6, 2)
        
        # Sum of coords: (N, 6, 2)
        pts = pos + verts
        
        # Dot with normals: (N, 6, 6)
        # normals: (1, 1, 6)
        dots = np.einsum('ijk,k->ij', pts, normals)
        
        max_proj = np.max(dots)
        
        # S * sqrt(3)/2 >= max_proj  => S >= 2/sqrt(3) * max_proj
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S

    def objective(params):
        positions = params.reshape(N, 2)
        return calc_S(positions)

    # Initial guess: Hexagonal lattice
    # Center at 0,0
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 2*sqrt(3)
    
    # Generate lattice points
    # Hex grid coordinates (i, j)
    # x = sqrt(3) * (i + j/2)
    # y = 1.5 * j
    
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # 6 neighbors (i,j) in range 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            if abs(i) + abs(j) > 1: continue # Only 6 neighbors
            x = np.sqrt(3) * (i + j/2.0)
            y = 1.5 * j
            points.append([x, y])
            
    # We have 7 points. Need 4 more.
    # Next shell: abs(i) + abs(j) == 2
    # There are 12 points in this shell. We pick 4.
    # To be symmetric, pick 4 that are far apart? Or just the first 4?
    # Let's pick 4 that maximize symmetry or just let the optimizer decide.
    # For initial guess, pick 4 from the 12.
    # Indices: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (0,-3)... wait, hex grid distance.
    # Distance metric in hex grid: max(|i|, |j|, |i+j|) ? No.
    # Let's just generate a few and pick 4.
    
    shell2 = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Check if in shell 2
            # Simple check: distance in cartesian approx
            x = np.sqrt(3) * (i + j/2.0)
            y = 1.5 * j
            r = np.sqrt(x*x + y*y)
            if 3.5 < r < 4.5: # Approx 2*sqrt(3) ~ 3.46
                 shell2.append([x, y])
                 
    # Just take 4 arbitrary ones from shell2 for initial guess
    # Better: take ones that are roughly 90 deg apart
    # Or just take the first 4. The optimizer will move them.
    if len(shell2) >= 4:
        points.extend(shell2[:4])
    else:
        # Fallback if generation failed
        for k in range(4):
            ang = k * np.pi/2
            points.append([2*np.sqrt(3)*np.cos(ang), 2*np.sqrt(3)*np.sin(ang)])

    initial_positions = np.array(points)
    
    # Optimization
    best_params = initial_positions.flatten()
    best_S = objective(best_params)
    
    # Use L-BFGS-B or similar
    # Objective is non-smooth (max), but L-BFGS often handles it well enough for this scale.
    # Or use a smooth approximation: log-sum-exp
    def smooth_objective(params):
        positions = params.reshape(N, 2)
        pos = positions[:, np.newaxis, :]
        verts = hex_verts[np.newaxis, :, :]
        pts = pos + verts
        dots = np.einsum('ijk,k->ij', pts, normals)
        
        # Soft max
        temp = 0.1
        max_dots = np.max(dots, axis=(1,2))
        # Actually, we want max over ALL dots
        all_dots = dots.flatten()
        max_d = np.max(all_dots)
        sum_exp = np.sum(np.exp((all_dots - max_d) / temp))
        soft_max = max_d + temp * np.log(sum_exp)
        
        S = (2.0 / np.sqrt(3.0)) * soft_max
        return S

    # Try optimization
    try:
        res = minimize(smooth_objective, best_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            best_params = res.x
            best_S = objective(best_params) # Recalc exact S
    except Exception:
        pass

    # Final check and formatting
    final_positions = best_params.reshape(N, 2)
    final_S = objective(final_positions)
    
    # Return format:
    # inner: list of [x, y, angle]
    # center: (0,0)
    # side: final_S
    # angle: 0
    
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i][0]), float(final_positions[i][1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
