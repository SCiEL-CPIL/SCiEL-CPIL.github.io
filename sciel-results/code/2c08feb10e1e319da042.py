# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _min_area_ratio_from_bary(bary):
    """
    bary: (N, 2) array of (u, v). w is derived.
    Returns the minimum area ratio of any triangle formed by 3 points.
    Area ratio = |det(P_i, P_j, P_k)| where P_k = (u_k, v_k, w_k).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Check validity: u,v,w >= -eps
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0 # Invalid, return negative score
    
    # Build (N, 3) matrix
    P = np.column_stack((u, v, w))
    
    min_det = 1.0
    # Iterate over all combinations of 3 indices
    # N=11, 165 combos. Vectorized is better.
    indices = np.arange(N)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    Pi = P[i_flat]
    Pj = P[j_flat]
    Pk = P[k_flat]
    
    # Determinant of 3x3 matrix formed by Pi, Pj, Pk as rows
    # det = Pi0*(Pj1*Pk2 - Pj2*Pk1) - Pi1*(Pj0*Pk2 - Pj2*Pk0) + Pi2*(Pj0*Pk1 - Pj1*Pk0)
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    min_det = np.min(np.abs(det))
    return min_det

def _objective_bary(x):
    """
    Objective function for minimization.
    x: flat array of length 2*N, containing u_i, v_i.
    """
    bary = x.reshape(N, 2)
    
    # Penalty for being outside the triangle
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # We want u, v, w >= 0
    penalty = 0.0
    if np.any(u < 0):
        penalty += np.sum(-u[u < 0])**2
    if np.any(v < 0):
        penalty += np.sum(-v[v < 0])**2
    if np.any(w < 0):
        penalty += np.sum(-w[w < 0])**2
        
    if penalty > 1e-6:
        return penalty * 100.0 # Strong penalty
        
    ratio = _min_area_ratio_from_bary(bary)
    if ratio < 0:
        return 100.0
    return -ratio

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    # If u+v > 1, scale down
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    # Create a grid of candidate points in barycentric space
    # u, v >= 0, u+v <= 1
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    # Start with the centroid
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        # Calculate distance from each candidate to the nearest selected point
        # Distance in barycentric space is Euclidean
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        # Select the candidate with max min-distance
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _structured_init_grid():
    """
    Simple grid initialization.
    """
    # 3x3 grid roughly
    # Points:
    # (1/3, 1/3)
    # (0.5, 0.1) ...
    # Let's just use FPS result as primary, and maybe a few random ones
    
    # Alternative: Centroids of 11 sub-triangles from a specific partition
    # This is hard to generate programmatically without geometry libraries.
    # FPS is a good proxy for "well spread".
    
    return _structured_init_fps()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    # Generate initial configurations
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. "Magic" Init: Vertices, Edge Midpoints, Centroid + 4 more
    # V0=(1,0,0) -> (1,0)
    # V1=(0,1,0) -> (0,1)
    # V2=(0,0,1) -> (0,0) in (u,v) where w=1-u-v? 
    # Wait, my mapping:
    # u corresponds to V0?
    # P = u*V0 + v*V1 + w*V2
    # If P=V0, u=1, v=0, w=0.
    # If P=V1, u=0, v=1, w=0.
    # If P=V2, u=0, v=0, w=1.
    
    magic_pts = [
        [1.0, 0.0], # V0
        [0.0, 1.0], # V1
        [0.0, 0.0], # V2
        [1/3, 1/3], # Centroid
        [0.5, 0.0], # Mid V0-V1
        [0.0, 0.5], # Mid V1-V2
        [0.5, 0.5], # Mid V0-V2? No, 0.5+0.5=1, w=0. Mid V0-V2 is u=0.5, v=0, w=0.5?
                    # P = 0.5 V0 + 0 V1 + 0.5 V2. Yes.
        # Need 4 more. Let's add quarter points.
        [0.25, 0.0], # 1/4 V0 - 3/4 V2? No. u=0.25, v=0, w=0.75.
        [0.0, 0.25], # u=0, v=0.25, w=0.75.
        [0.25, 0.25], # u=0.25, v=0.25, w=0.5.
        [0.5, 0.25], # u=0.5, v=0.25, w=0.25.
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # Run Nelder-Mead on each init
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Project init to valid simplex
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            res = minimize(
                _objective_bary,
                init_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            x_opt = res.x.reshape(N, 2)
            x_opt = _project_to_bary_simplex(x_opt)
            
            score = _min_area_ratio_from_bary(x_opt)
            if score > best_score:
                best_score = score
                best_x = x_opt
        except Exception:
            pass
            
    # If we have time, do a few more random restarts or DE?
    # Let's just do a few more NM restarts if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline:
                break
            pts = np.zeros((N, 2))
            for i in range(N):
                u = np.random.random()
                v = np.random.random()
                while u + v > 1.0:
                    u = np.random.random()
                    v = np.random.random()
                pts[i, 0] = u
                pts[i, 1] = v
            pts = _project_to_bary_simplex(pts)
            try:
                res = minimize(
                    _objective_bary,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                x_opt = res.x.reshape(N, 2)
                x_opt = _project_to_bary_simplex(x_opt)
                score = _min_area_ratio_from_bary(x_opt)
                if score > best_score:
                    best_score = score
                    best_x = x_opt
            except Exception:
                pass

    if best_x is None:
        # Fallback: FPS
        best_x = _structured_init_fps()
        
    # Convert back to Cartesian for output
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
