"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0 # Safety margin from 150s
MIN_DIST = np.sqrt(3) # Minimum center-to-center distance for disjoint unit hexagons

def _objective(params):
    """
    params: [x1, y1, theta1, x2, y2, theta2, ...]
    Returns the side length S of the minimal enclosing regular hexagon (flat-top).
    Includes penalty for overlapping (distance < MIN_DIST).
    """
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    # 1. Compute vertices of all inner hexagons
    # Shape: (N, 6, 2)
    th = np.deg2rad(ang)[:, None] + np.arange(6) * np.pi / 3.0
    c = np.cos(th)
    s = np.sin(th)
    xs = cx[:, None] + c
    ys = cy[:, None] + s
    verts = np.stack((xs, ys), axis=-1)
    
    # 2. Compute projections onto 6 normals of the outer hexagon
    # Outer hexagon is flat-top, centered at (0,0).
    # Normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Normal vectors:
    # n0: (cos30, sin30)
    # n1: (cos90, sin90)
    # ...
    normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    
    # Project all vertices (N*6, 2) onto normals (6, 2)
    # Flatten verts
    v_flat = verts.reshape(-1, 2)
    # Projections shape: (N*6, 6)
    projs = v_flat @ normals.T
    # Max absolute projection
    max_proj = np.max(np.abs(projs))
    
    # Side length S = max_proj / (sqrt(3)/2)
    S = 2.0 * max_proj / np.sqrt(3.0)
    
    # 3. Penalty for overlap (distance constraint)
    # Calculate pairwise distances
    # cx, cy shape (N,)
    # dist matrix shape (N, N)
    diff = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dists = np.sqrt(diff**2 + diff_y**2)
    
    # Penalty: sum of (MIN_DIST - dist)^2 for dist < MIN_DIST
    # Mask upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    violations = np.maximum(0.0, MIN_DIST - d_ij)
    penalty = np.sum(violations**2)
    
    return S + 100.0 * penalty

def construct():
    t0 = time.time()
    
    # Initial Guess: Hexagonal lattice subset
    # 1 center, 6 neighbors, 5 more neighbors
    # Lattice vectors for hexagonal packing of side 1:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 1.5)
    # Distance between centers is sqrt(3).
    
    # Let's generate points in a hexagonal grid
    # Center (0,0)
    # Ring 1: 6 points at distance sqrt(3)
    # Ring 2: 12 points at distance 2*sqrt(3)
    # We need 12 points total.
    
    # Points:
    # (0,0)
    # Ring 1:
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    # That's 7 points.
    # We need 5 more.
    # Pick 5 from Ring 2.
    # Ring 2 points (examples):
    # (2*sqrt(3), 0)
    # (3*sqrt(3)/2, 1.5)
    # (sqrt(3), 3)
    # (0, 3)
    # (-sqrt(3), 3)
    # ...
    
    # Let's just use a simple grid or random start if this is too complex.
    # Actually, a "flower" of 7 + 5 outer is good.
    
    centers = []
    angles = []
    
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    
    # Ring 1 (6)
    for k in range(6):
        a = k * np.pi / 3
        x = np.sqrt(3) * np.cos(a)
        y = np.sqrt(3) * np.sin(a)
        centers.append((x, y))
        # Orient them to point outward? Or just 0.
        angles.append(0.0)
        
    # Ring 2 (5 selected)
    # To keep it compact, pick 5 consecutive ones from Ring 2?
    # Or just randomize a bit and let optimizer fix it.
    # Let's pick 5 points from Ring 2 that are "gaps" or extensions.
    # Actually, just adding 5 more points at distance 2*sqrt(3) in random directions
    # and letting the optimizer push them into place is robust.
    
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi/5 # Offset
        x = 2 * np.sqrt(3) * np.cos(a)
        y = 2 * np.sqrt(3) * np.sin(a)
        centers.append((x, y))
        angles.append(0.0)
        
    x0 = []
    for i in range(N):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
        
    x0 = np.array(x0)
    
    # Optimize
    # Use Powell for robustness with non-smooth max function
    # Nelder-Mead is also good.
    
    best_x = x0
    best_val = _objective(x0)
    
    # Run optimization
    # Limit iterations to ensure time safety
    # Powell can be slow if not careful.
    
    try:
        # Use Nelder-Mead first, it's often faster for small N
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Try Powell if time permits
        if time.time() - t0 < DEADLINE - 20:
            res2 = minimize(_objective, best_x, method='Powell',
                            options={'maxiter': 1000, 'xtol': 1e-4, 'ftol': 1e-6})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception:
        pass
    
    # Extract result
    cx = best_x[0::3]
    cy = best_x[1::3]
    ang = best_x[2::3]
    
    # Calculate final S
    # Re-calculate S from best_x to be sure
    S = _objective(best_x)
    # The objective includes penalty. If penalty is 0, S is the answer.
    # If penalty is not 0, the solution is invalid.
    # But the optimizer should drive penalty to 0.
    # To get the raw S, we can re-compute without penalty or just return the objective if penalty is small.
    # The task says "score is the outer side length".
    # If the hexagons overlap, the packing is invalid.
    # We should ensure the returned configuration is valid.
    # If penalty > 1e-3, we might have a problem.
    # However, the initial guess is valid (distances >= sqrt(3) approximately).
    # The optimizer minimizes S + Penalty.
    # If it finds a local minimum with overlap, S might be small but invalid.
    # We should check the penalty.
    
    # Let's re-evaluate the penalty separately
    diff = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dists = np.sqrt(diff**2 + diff_y**2)
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    min_dist = np.min(d_ij)
    
    if min_dist < MIN_DIST - 1e-3:
        # Fallback to a known valid configuration if optimization failed to separate them
        # Use the initial guess which was valid
        cx = x0[0::3]
        cy = x0[1::3]
        ang = x0[2::3]
        # Recalculate S for this valid config
        # ... (re-use logic or just return a large S)
        # For now, let's assume the optimizer works.
        # But to be safe, if invalid, return a large side length.
        return [[float(cx[i]), float(cy[i]), float(ang[i])] for i in range(N)], (0.0, 0.0), 50.0, 0.0

    # Calculate S for the valid configuration
    th = np.deg2rad(ang)[:, None] + np.arange(6) * np.pi / 3.0
    c = np.cos(th)
    s = np.sin(th)
    xs = cx[:, None] + c
    ys = cy[:, None] + s
    verts = np.stack((xs, ys), axis=-1)
    
    normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    
    v_flat = verts.reshape(-1, 2)
    projs = v_flat @ normals.T
    max_proj = np.max(np.abs(projs))
    S = 2.0 * max_proj / np.sqrt(3.0)
    
    inner = [[float(cx[i]), float(cy[i]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0

