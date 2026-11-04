import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate optimal spacing. 
    # Area of unit circle = pi. Area of 300 circles = 300 * pi * r^2.
    # Density of hex packing ~ 0.9069.
    # 300 * pi * r^2 / pi = 300 * r^2.
    # 300 * r^2 / 0.9069 <= 1  =>  r <= sqrt(1 / (300 * 0.9069)) ≈ sqrt(1/272) ≈ 0.0607.
    # This is an upper bound. The boundary effects reduce it.
    # Known best is ~0.053.
    # Let's start with a spacing that allows for adjustment.
    # d = 2r. Start with r ~ 0.05. d ~ 0.1.
    
    # Generate a dense hex grid
    d_init = 0.10 
    points = []
    # Hex lattice vectors
    v1 = np.array([d_init, 0])
    v2 = np.array([d_init * 0.5, d_init * np.sqrt(3) / 2])
    
    # Range to cover unit circle
    R = 1.5
    n_max = int(R / d_init) + 2
    
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            p = i * v1 + j * v2
            if np.linalg.norm(p) <= 1.0:
                points.append(p)
                
    points = np.array(points)
    
    # If we have too many, pick the N closest to center
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # If not enough, reduce spacing and retry
        # This is unlikely with d=0.1, but just in case
        d_init = 0.09
        points = []
        v1 = np.array([d_init, 0])
        v2 = np.array([d_init * 0.5, d_init * np.sqrt(3) / 2])
        n_max = int(R / d_init) + 2
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                p = i * v1 + j * v2
                if np.linalg.norm(p) <= 1.0:
                    points.append(p)
        points = np.array(points)
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx[:N]]
            
    # 2. Optimization: Force Directed Relaxation
    # We want to maximize the minimum distance.
    # We will use a gradient-based approach on a smooth surrogate.
    # Objective: Maximize soft-min of distances.
    # Let d_ij be distance between i and j.
    # Let d_i be distance to boundary (1 - |c_i|).
    # We want to maximize min(all d_ij, all d_i).
    # Surrogate: S = (prod d_ij * prod d_i)^(1/M)  => Maximize log S = (1/M) * (sum log d_ij + sum log d_i)
    # However, log is problematic if d goes to 0.
    # Better: Maximize sum of (d_ij - r)^2 for all pairs where d_ij < 2r? No, r is variable.
    
    # Let's use the "Repulsive Force" model.
    # Each pair (i,j) exerts a force if they are closer than a target distance.
    # But what is the target distance? It's the variable we want to maximize.
    
    # Alternative: Simple Gradient Ascent on the "Soft Min".
    # f(x) = -log( sum_{i<j} exp(-beta * d_ij^2) + sum_i exp(-beta * (1-|c_i|)^2) )
    # Gradient of f w.r.t c_k involves terms like 2*beta*exp(-beta*d^2)*d_vec / sum_exp.
    
    beta = 50.0
    lr = 0.001
    steps = 5000
    
    # Precompute indices for pairs? N=300, N^2/2 = 45000 pairs.
    # Vectorized numpy operations are fast enough for 5000 steps?
    # 5000 * 45000 = 225M ops. Might be tight in 100s.
    # Let's use a spatial hash or just rely on N=300 being small enough for dense matrix ops if optimized.
    # 300x300 matrix is 90k elements. Very small.
    
    def compute_forces(pts):
        # pts: (N, 2)
        # Return gradient of the objective to be maximized.
        
        # Pairwise distances
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        # dist_sq = np.sum(diff**2, axis=2) # (N, N)
        
        # To save memory/time, we can do it in blocks or just use the full matrix.
        # N=300 is small.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Boundary distances
        radii = np.linalg.norm(pts, axis=1)
        boundary_dist = 1.0 - radii
        boundary_dist_sq = boundary_dist**2
        
        # Objective term: sum exp(-beta * dist_sq)
        # We want to maximize the min, so we minimize the sum of exp(-beta * dist_sq) is wrong.
        # If dist is small, exp(-beta*dist^2) is large.
        # We want to push apart small distances.
        # So we want to MINIMIZE the sum of exp(-beta * dist_sq) for pairs?
        # No, if dist increases, exp(-beta*dist^2) decreases.
        # So minimizing the sum of exp(-beta*dist^2) encourages larger distances.
        # This is a "soft" repulsion.
        
        # Let's define the loss L = sum_{i<j} exp(-beta * d_ij^2) + sum_i exp(-beta * b_i^2)
        # We want to minimize L.
        
        # Gradient of L w.r.t c_k:
        # d/dc_k sum_{j != k} exp(-beta * |c_k - c_j|^2)
        # = sum_{j != k} exp(-beta * d_kj^2) * (-2 * beta * (c_k - c_j))
        
        # d/dc_k exp(-beta * (1 - |c_k|)^2)
        # Let r_k = |c_k|. b_k = 1 - r_k.
        # d/dc_k = exp(-beta * b_k^2) * (-2 * beta * b_k) * d/dc_k (1 - r_k)
        # d/dc_k r_k = c_k / r_k
        # d/dc_k (1 - r_k) = - c_k / r_k
        # So term = exp(-beta * b_k^2) * (-2 * beta * b_k) * (- c_k / r_k)
        # = 2 * beta * b_k * exp(-beta * b_k^2) * (c_k / r_k)
        
        # Total Gradient = Sum of pair gradients + Boundary gradient
        
        # Pair Gradients
        # G_pair[k] = sum_{j != k} exp(-beta * dist_sq[k,j]) * (-2 * beta * (c_k - c_j))
        # Note: diff[k,j] = c_k - c_j.
        # exp_mat = np.exp(-beta * dist_sq)
        # np.fill_diagonal(exp_mat, 0)
        # G_pair = np.sum(exp_mat[:, np.newaxis, :] * diff * (-2 * beta), axis=1)
        # Wait, diff is (N,N,2). exp_mat is (N,N).
        # We need to broadcast exp_mat to (N,N,1).
        
        exp_mat = np.exp(-beta * dist_sq)
        np.fill_diagonal(exp_mat, 0.0)
        
        # Weighted sum of differences
        # G_pair = -2 * beta * np.sum(exp_mat[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary Gradients
        # G_bound[k] = 2 * beta * b_k * exp(-beta * b_k^2) * (c_k / r_k)
        # Handle r_k = 0
        safe_r = np.where(radii > 1e-8, radii, 1.0)
        unit_vec = pts / safe_r[:, np.newaxis]
        # If r is 0, unit_vec is 0/1 = 0, which is fine (gradient 0)
        
        coeff = 2.0 * beta * boundary_dist * np.exp(-beta * boundary_dist_sq)
        G_bound = coeff[:, np.newaxis] * unit_vec
        
        return -2.0 * beta * np.sum(exp_mat[:, :, np.newaxis] * diff, axis=1) + G_bound

    # Optimization Loop
    pts = points.copy()
    best_pts = pts.copy()
    best_score = -np.inf
    
    # Compute initial score
    def score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq)) / 2.0
        
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = 1.0 - np.max(radii)
        
        return min(min_pair_dist, min_boundary_dist)
    
    current_score = score(pts)
    if current_score > best_score:
        best_score = current_score
        best_pts = pts.copy()
        
    for step in range(steps):
        if time.time() > deadline:
            break
            
        grad = compute_forces(pts)
        
        # Adaptive step size?
        # Simple gradient ascent
        pts += lr * grad
        
        # Check score every few steps
        if step % 10 == 0:
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                
        # Reduce learning rate slightly
        lr *= 0.999
        
    return best_pts
