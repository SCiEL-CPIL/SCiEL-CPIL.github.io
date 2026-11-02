"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # safety margin from 150s limit

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _cartesian_to_barycentric(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    # w = y / S3H
    w = y / S3H
    # u = 1 - x - 0.5*w
    u = 1.0 - x - 0.5 * w
    # v = x - 0.5*w
    v = x - 0.5 * w
    return np.column_stack([u, v, w])

def _soft_min_objective_flat(flat_params, alpha=100.0):
    """
    Objective in barycentric coordinates.
    flat_params: 2*N array of (u_i, v_i) for i=0..N-1, where w_i = 1 - u_i - v_i.
    We maximize min area, so minimize -soft_min(areas).
    Add penalty for points outside triangle.
    """
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for negative barycentric coords (outside triangle)
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    # Convert to Cartesian
    x = v + 0.5 * w
    y = S3H * w
    
    # Compute all triangle areas
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    # Soft minimum: -log(sum(exp(-alpha * areas))) / alpha
    # This is differentiable and approximates the min
    min_val = -np.log(np.sum(np.exp(-alpha * valid_areas))) / alpha
    return -min_val + penalty

def _min_area_objective_flat(flat_params):
    """Non-smooth objective: negative min area plus penalty."""
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    x = v + 0.5 * w
    y = S3H * w
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    min_val = np.min(valid_areas) / TRI_AREA
    return -min_val + penalty

def _is_valid(points):
    """Check if all points are inside the triangle."""
    bary = _cartesian_to_barycentric(points)
    return np.all(bary >= -1e-9)

def _farthest_point_sampling(rng, n_points):
    """
    Generate n_points in the triangle using Farthest Point Sampling (FPS).
    This provides well-spread initial guesses.
    """
    points = []
    # Start with a random point
    while True:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            break
    points.append(np.array([u + 0.5 * v, S3H * v]))
    
    for _ in range(n_points - 1):
        # Compute min distance to existing points for a grid of candidates
        # To be fast, we sample a grid and pick the one with max min-distance
        best_dist = -1
        best_pt = None
        
        # Sample candidates
        for _ in range(200):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    break
            cand = np.array([u + 0.5 * v, S3H * v])
            
            # Compute min distance to existing points
            dists = np.linalg.norm(points - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        points.append(best_pt)
        
    return np.array(points)

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initial guesses using FPS and random
    initial_guesses = []
    
    # FPS-based start
    fps_pts = _farthest_point_sampling(rng, N)
    bary = _cartesian_to_barycentric(fps_pts)
    initial_guesses.append(np.column_stack([bary[:, 0], bary[:, 1]]).flatten())
    
    # Random starts
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        s[s == 0] = 1.0
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        initial_guesses.append(np.column_stack([u, v]).flatten())
    
    # Phase 2: Powell optimization from each initial guess
    # Powell is robust for non-smooth objectives
    for i, x0 in enumerate(initial_guesses):
        if elapsed() > DEADLINE - 20:
            break
        
        def obj_powell(x):
            return _min_area_objective_flat(x)
        
        try:
            res = minimize(
                obj_powell,
                x0,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 15:
            break
        
        # Convert back to Cartesian
        uv = res.x.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Check validity
        if np.min(u) < -1e-6 or np.min(v) < -1e-6 or np.min(w) < -1e-6:
            # Clip and re-evaluate
            u = np.maximum(u, 0)
            v = np.maximum(v, 0)
            w = 1.0 - u - v
            if np.min(w) < -1e-6:
                continue
        
        x_cart = v + 0.5 * w
        y_cart = S3H * w
        points = np.column_stack([x_cart, y_cart])
        
        if not _is_valid(points):
            continue
        
        score = _min_area_normalized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 3: Polish the best candidate with L-BFGS-B using soft-min objective
    if best_points is not None and elapsed() < DEADLINE - 5:
        bary = _cartesian_to_barycentric(best_points)
        x0 = np.column_stack([bary[:, 0], bary[:, 1]]).flatten()
        
        def obj_refine(x):
            return _soft_min_objective_flat(x, alpha=200.0)
        
        try:
            res = minimize(
                obj_refine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            
            if np.min(u) >= -1e-6 and np.min(v) >= -1e-6 and np.min(w) >= -1e-6:
                x_cart = v + 0.5 * w
                y_cart = S3H * w
                points = np.column_stack([x_cart, y_cart])
                
                if _is_valid(points):
                    score = _min_area_normalized(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish in Cartesian space
    if best_points is not None and elapsed() < DEADLINE - 2:
        def cart_objective(flat):
            pts = flat.reshape(N, 2)
            bary = _cartesian_to_barycentric(pts)
            penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                               np.sum(np.maximum(0, -bary[:, 1])**2) +
                               np.sum(np.maximum(0, -bary[:, 2])**2))
            min_a = _min_area_normalized(pts)
            return -min_a + penalty
        
        try:
            res = minimize(
                cart_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished = res.x.reshape(N, 2)
            if _is_valid(polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Fallback if no valid points found
    if best_points is None:
        # Return FPS points as fallback
        best_points = _farthest_point_sampling(rng, N)
    
    return best_points
