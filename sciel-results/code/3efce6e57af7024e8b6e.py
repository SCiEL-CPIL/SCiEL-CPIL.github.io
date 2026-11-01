"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Initial positions
    # 1 center
    # 6 ring at distance SQRT3
    # 4 extra at distance 2*SQRT3, angles 30, 90, 150, 270
    
    angles_ring = np.arange(6) * np.pi / 3.0
    positions = [(0.0, 0.0)]
    for a in angles_ring:
        positions.append((SQRT3 * np.cos(a), SQRT3 * np.sin(a)))
    
    # 4 extra
    # Place them in the "gaps" of the next shell
    # Angles: pi/6, pi/2, 5pi/6, 3pi/2
    extra_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 3*np.pi/2]
    for a in extra_angles:
        positions.append((2*SQRT3 * np.cos(a), 2*SQRT3 * np.sin(a)))
        
    x0 = np.array([coord for p in positions for coord in p])
    
    # Normals for the container hexagon (pointy top? or flat top?)
    # Standard: Flat top means normals at 0, 60, 120...
    # Let's use normals at 0, pi/3, ...
    normals = np.array([
        [np.cos(k*np.pi/3.0), np.sin(k*np.pi/3.0)] for k in range(6)
    ])
    
    # Vertices of a unit hexagon (side 1, center 0,0, rotation 0)
    # Distance to vertex is 1.
    # Angles: 0, pi/3, ...
    hex_verts = np.array([
        [np.cos(k*np.pi/3.0), np.sin(k*np.pi/3.0)] for k in range(6)
    ])
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(N, 2)
        
        # Calculate max projection of all vertices onto all normals
        max_proj = 0.0
        for i in range(N):
            center = pos[i]
            # All vertices of hex i
            verts = center + hex_verts
            # Projections onto normals
            # verts @ normals.T gives 6x6 matrix
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
            
        # Side length s is related to max_proj by max_proj = s * SQRT3 / 2
        s = 2.0 * max_proj / SQRT3
        
        # Penalty for overlaps
        # Min distance SQRT3
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < SQRT3:
                    penalty += (SQRT3 - d)**2
        
        return s + 100.0 * penalty

    # Optimization
    # L-BFGS-B is good for this
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500})
    
    if time.time() > deadline:
        # Fallback to initial or best found
        # If res is not valid, use x0
        if res.success or res.fun < 1e9:
            best_x = res.x
        else:
            best_x = x0
    else:
        best_x = res.x

    # Extract result
    pos = best_x.reshape(N, 2)
    
    # Calculate final side length
    max_proj = 0.0
    for i in range(N):
        center = pos[i]
        verts = center + hex_verts
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    final_side = 2.0 * max_proj / SQRT3
    
    # Check validity (no overlaps)
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pos[i] - pos[j]) < SQRT3 - 1e-5:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, return a known valid configuration (initial)
        # This shouldn't happen if penalty works, but safety first
        pos = x0.reshape(N, 2)
        max_proj = 0.0
        for i in range(N):
            center = pos[i]
            verts = center + hex_verts
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
        final_side = 2.0 * max_proj / SQRT3

    inner = [[p[0], p[1], 0.0] for p in pos]
    return inner, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
