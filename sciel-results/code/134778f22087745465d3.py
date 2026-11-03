"""
Strategy: Spectral Uniformity + Fast Local Refinement
Instead of relying on ConvexHull (which is slow and numerically unstable for 
optimization loops), we use two stages:
1. Spectral Force-Directed Method: Minimizes the variance of pairwise distances 
   by adjusting points to reduce the largest eigenvalue of the Gram matrix (excluding 
   the trivial eigenvalue). This pushes points towards a uniform distribution.
2. Grid-based Local Refinement: Uses a pre-computed dense grid of points on the sphere.
   For each grid point, we find the nearest configuration point. The "error" is the 
   distance. We move each configuration point towards the centroid of the grid points 
   for which it is the nearest neighbor (Voronoi-like). This directly minimizes the 
   covering radius approximation.
"""

import numpy as np
import time
import math

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _covering_radius(pts, grid_pts):
    """
    Estimate covering radius using a dense grid.
    grid_pts: (M, 3) array of unit vectors.
    Returns the maximum angular distance from any grid point to its nearest pts.
    """
    # Compute dot products: (M, N)
    # dists[i, j] = arccos(grid_pts[i] . pts[j])
    # To avoid arccos cost, we can maximize the minimum dot product? No, minimize max distance = maximize min dot product?
    # min distance = arccos(max_dot).
    # So we want to maximize the minimum over i of (max over j of dot(grid_i, pts_j)).
    
    dots = grid_pts @ pts.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    # Clip for numerical safety
    max_dots = np.clip(max_dots, -1.0, 1.0)
    min_dot = np.min(max_dots)
    radius_rad = np.arccos(min_dot)
    return np.degrees(radius_rad)

def _covering_radius_and_grad(pts, grid_pts):
    """
    Compute covering radius and gradient using the grid approximation.
    This is a differentiable approximation of the true covering radius.
    """
    dots = grid_pts @ pts.T  # (M, N)
    max_dots = np.max(dots, axis=1)  # (M,)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    
    # The radius is determined by the grid point with the smallest max_dot.
    # Let i* be argmin(max_dots).
    # The gradient of the radius w.r.t pts is dominated by the "worst" grid points.
    # We use a soft-max/soft-min to get a smooth gradient.
    
    # f = -logsumexp(beta * dots) / beta  approximates min_j dot(g_i, p_j)
    # We want to maximize min_i (max_j dot(g_i, p_j)).
    # Let h_i = max_j dot(g_i, p_j). We want to maximize min_i h_i.
    # Equivalent to minimizing -min_i h_i.
    
    # Approximate max_j dot(g_i, p_j) with logsumexp?
    # max_j x_j ≈ (1/beta) log sum exp(beta x_j)
    # Then min_i h_i ≈ (1/beta) log sum exp(-beta h_i) ... no, min is -max(-x).
    # min_i h_i = - max_i (-h_i) ≈ -(1/beta) log sum exp(-beta h_i).
    
    # This is getting complex. Let's use a simpler heuristic gradient.
    # Identify the "worst" grid points (those with low max_dot).
    # For each such grid point g, find the nearest point p_j.
    # The gradient of the distance d(g, p_j) w.r.t p_j is in the direction (g - p_j) projected to tangent.
    # We accumulate these forces.
    
    # Find indices of worst grid points (bottom 10% or so)
    k_worst = max(10, len(grid_pts) // 100)
    worst_indices = np.argpartition(max_dots, k_worst)[:k_worst]
    
    grad = np.zeros_like(pts)
    
    # For each worst grid point, find nearest pt and add force
    for i in worst_indices:
        g = grid_pts[i]
        # Find nearest pt
        j = np.argmax(dots[i])
        p = pts[j]
        
        # Force direction: move p towards g
        # Vector from p to g on sphere
        v = g - np.dot(g, p) * p
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-8:
            v /= norm_v
            # Weight by how "bad" this point is (lower dot is worse)
            weight = 1.0 - max_dots[i] # Roughly proportional to distance
            grad[j] += weight * v
            
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-8:
        grad /= grad_norm
        
    radius_deg = np.degrees(np.arccos(np.min(max_dots)))
    return radius_deg, grad

def _spectral_step(pts, lr=0.01):
    """
    Spectral force-directed step.
    Minimizes the variance of pairwise distances by reducing the largest non-trivial eigenvalue
    of the Gram matrix.
    """
    G = pts @ pts.T
    # Center G? No, for sphere packing, we look at the eigenvalues of G.
    # The largest eigenvalue corresponds to the mean direction (should be 0 if balanced).
    # We want to minimize the sum of squared deviations from the mean?
    # A common objective is to minimize the "energy" sum_{i<j} (d_ij - d_mean)^2.
    # Alternatively, minimize the largest eigenvalue of (G - I/2)?
    
    # Simpler: Force-directed repulsion.
    # Pairwise forces: if points are too close, repel. If too far, attract?
    # For uniform distribution, we want to maximize the minimum distance (packing) 
    # but also cover. 
    # The spectral method often minimizes the trace of G^2 or similar.
    # Let's use a simple repulsive force based on short-range interactions to spread points.
    
    # Compute all pairwise distances (chordal)
    # d_ij = ||p_i - p_j||^2 = 2 - 2 p_i.p_j
    # We want to avoid clusters.
    
    # Gradient of sum_{i<j} exp(-alpha * d_ij^2)
    # This creates repulsion.
    
    alpha = 2.0
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    
    # Forces
    # F_i = sum_{j != i} p_i - p_j * derivative
    # Let's use a simple heuristic: move points away from their nearest neighbors if they are too close.
    # This is effectively what the grid refinement does, but spectral is global.
    
    # For this step, we will just use a small random jitter + projection to break symmetry,
    # combined with the grid refinement which is more targeted.
    # Actually, let's implement a proper spectral gradient:
    # Minimize E = sum_{i,j} (p_i . p_j)^2
    # dE/dp_i = 4 * sum_j (p_i . p_j) p_j
    # This tends to orthogonalize points, which is good for uniformity.
    
    grad = 4.0 * (dots @ pts)
    
    # Project to tangent space
    grad -= np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    pts_new = pts + lr * grad
    return _normalize(pts_new)

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_grid(n_theta=180, n_phi=360):
    """
    Generate a dense grid of points on the sphere for covering radius estimation.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Pre-compute grid
    # Use a reasonably dense grid but not too dense to keep matrix ops fast
    # 100x200 = 20000 points. 20000x130 matrix is ~2.6M floats. Fast.
    grid_pts = _generate_grid(90, 180) # ~16200 points
    
    best_pts = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Fibonacci
    fib_pts = _fibonacci_spiral(N)
    candidates.append(fib_pts)
    
    # 2. Random
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        candidates.append(pts)
        
    # Evaluate and pick best start
    for pts in candidates:
        if not _time_ok(deadline, 20.0):
            break
        try:
            score, _ = _covering_radius(pts, grid_pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = fib_pts
        best_score = _covering_radius(best_pts, grid_pts)
        
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Phase 1: Spectral Uniformity (fast, global)
    # Run for a few seconds to get a good spread
    t_start_phase1 = time.time()
    lr_spec = 0.005
    while time.time() - t_start_phase1 < 5.0 and _time_ok(deadline, 30.0):
        current_pts = _spectral_step(current_pts, lr_spec)
        # Decay learning rate
        lr_spec *= 0.99
        # Periodically check score
        if int((time.time() - t_start_phase1) * 2) % 2 == 0: # ~every 0.5s
            score, _ = _covering_radius(current_pts, grid_pts)
            if score < best_score:
                best_score = score
                best_pts = current_pts.copy()
    
    # Phase 2: Grid-based Local Refinement (Lloyd-like)
    # This directly targets the covering radius
    t_start_phase2 = time.time()
    lr_lloyd = 0.1
    momentum = 0.5
    v = np.zeros_like(current_pts)
    
    no_improve = 0
    max_no_improve = 50
    
    while _time_ok(deadline, 10.0) and no_improve < max_no_improve:
        score, grad = _covering_radius_and_grad(current_pts, grid_pts)
        
        if not np.isfinite(score):
            break
            
        # Update velocity
        v = momentum * v + lr_lloyd * grad
        new_pts = _normalize(current_pts + v)
        
        new_score, _ = _covering_radius(new_pts, grid_pts)
        
        if new_score < current_score - 1e-6:
            current_pts = new_pts
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            no_improve += 1
            lr_lloyd *= 0.9
            v *= 0.5
            
    # Phase 3: Fine-tuning with smaller steps
    if _time_ok(deadline, 5.0):
        current_pts = best_pts.copy()
        current_score = best_score
        lr_lloyd = 0.01
        v = np.zeros_like(current_pts)
        no_improve = 0
        
        while _time_ok(deadline, 2.0) and no_improve < 20:
            score, grad = _covering_radius_and_grad(current_pts, grid_pts)
            if not np.isfinite(score):
                break
            v = 0.5 * v + lr_lloyd * grad
            new_pts = _normalize(current_pts + v)
            new_score, _ = _covering_radius(new_pts, grid_pts)
            
            if new_score < current_score - 1e-7:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve += 1
                lr_lloyd *= 0.9
                v *= 0.5

    return best_pts
