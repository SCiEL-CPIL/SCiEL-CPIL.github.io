import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # 1. Initialize with Hexagonal Lattice
    # We want to fit N points in [-0.5, 0.5]^2
    # Hex lattice basis:
    # v1 = (2r, 0)
    # v2 = (r, r*sqrt(3))
    # We don't know r yet. Let's estimate r from density.
    # rho = pi / (2*sqrt(3))
    # r_est = sqrt(rho / (N * pi)) * 1.0 # Area = 1
    # Let's just generate a grid and scale it.
    
    # Generate hex points in a large enough area, then scale to fit [-0.5, 0.5]
    # Number of rows/cols to get ~200 points
    # Hex packing: rows are offset.
    # Let's try to generate exactly N points or close.
    
    # Simpler: Generate a square grid of hex points, then trim/scale.
    # Or, just use a square grid of "hex-like" spacing?
    # No, let's create a proper hex grid.
    
    # Estimate r
    r_est = np.sqrt(1.0 / (N * np.sqrt(3))) # approx
    
    # Create lattice vectors
    # Spacing a = 2*r
    a = 2 * r_est
    dx = a
    dy = a * np.sqrt(3) / 2
    
    # Generate points
    # We need enough points to cover the square and then crop/scale
    # Let's generate a grid of size M x M
    M = int(np.ceil(np.sqrt(N) * 1.5)) # Oversample
    
    xs = []
    ys = []
    for row in range(M):
        for col in range(M):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            xs.append(x)
            ys.append(y)
            
    pts = np.column_stack((xs, ys))
    
    # Center the points
    pts -= pts.mean(axis=0)
    
    # Scale to fit in [-0.5, 0.5]
    # Find max extent
    max_ext = np.max(np.abs(pts))
    if max_ext > 0:
        pts /= max_ext
        pts *= 0.5 # Fit to half-side 0.5
    
    # We now have M*M points. We need N.
    # Select the N points closest to the center? Or just the first N?
    # Better: Select N points that are well distributed.
    # Since they are on a lattice, we can just take the first N if generated in a compact way?
    # My generation above is a square block of hex lattice.
    # Let's just take the first N points.
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points if needed (unlikely)
        while len(pts) < N:
            pts = np.vstack([pts, np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])])
            
    pts = pts.astype(np.float64)
    
    # 2. Optimize using Repulsive Gradient Descent
    # Objective: Minimize E = sum_{i<j} exp(-alpha * d_ij)
    # This pushes close points apart.
    
    alpha = 50.0 # Sharpness
    lr = 0.01    # Learning rate
    momentum = 0.9
    v = np.zeros_like(pts)
    
    best_score = -np.inf
    best_pts = pts.copy()
    
    # Precompute indices for pairs
    # N=200, pairs = 19900.
    # We can vectorize the gradient calculation.
    
    def get_score(p):
        # Calculate min distance to other points and boundaries
        # Distances between all pairs
        # p is (N, 2)
        # Use broadcasting
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        # p is in [-0.5, 0.5]
        # dist to boundary = min(0.5 - x, x + 0.5) for each coord
        # Actually, min(0.5 - |x|)
        dist_to_boundary = 0.5 - np.abs(p)
        min_boundary_dist = np.min(dist_to_boundary)
        
        # The radius is min(min_pair_dist / 2, min_boundary_dist)
        return min(min_pair_dist / 2, min_boundary_dist)

    best_score = get_score(pts)
    
    for step in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate Gradient
        # E = sum_{i<j} exp(-alpha * d_ij)
        # dE/dx_i = sum_{j != i} alpha * exp(-alpha * d_ij) * (x_i - x_j) / d_ij
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        
        # We only need i < j for the sum, but gradient for i involves all j.
        # Let's compute the force on each point i from all j.
        # Force magnitude factor: alpha * exp(-alpha * d) / d
        # Direction: (x_i - x_j)
        
        # exp_term = np.exp(-alpha * dists_safe)
        # This matrix is N x N.
        # Force on i: sum_j alpha * exp(-alpha * d_ij) * (x_i - x_j) / d_ij
        # Note: d_ij = d_ji. exp is symmetric.
        
        # To save memory/time, we can iterate or use sparse?
        # N=200 is small enough for dense.
        
        exp_term = np.exp(-alpha * dists_safe)
        # Set diagonal to 0
        np.fill_diagonal(exp_term, 0)
        
        # Factor for each pair (i,j) in the sum for point i:
        # term_ij = alpha * exp_term[i,j] / dists_safe[i,j]
        # grad_i = sum_j term_ij * (pts[i] - pts[j])
        
        # Vectorized:
        # grad = sum_j [ (alpha * exp_term / dists_safe) * (pts[i] - pts[j]) ]
        # Let W[i,j] = alpha * exp_term[i,j] / dists_safe[i,j]
        W = alpha * exp_term / dists_safe
        np.fill_diagonal(W, 0)
        
        # grad[i] = sum_j W[i,j] * (pts[i] - pts[j])
        # = pts[i] * sum_j W[i,j] - sum_j W[i,j] * pts[j]
        
        sum_W = np.sum(W, axis=1)
        # W is (N, N), pts is (N, 2)
        # sum_j W[i,j] * pts[j] is W @ pts
        weighted_sum_pts = W @ pts
        
        grad = pts * sum_W[:, np.newaxis] - weighted_sum_pts
        
        # Update with momentum
        v = momentum * v - lr * grad
        pts += v
        
        # Project to box [-0.5, 0.5]
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check score occasionally
        if step % 100 == 0:
            score = get_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        # Adaptive learning rate?
        # If score improves, keep lr. If not, reduce.
        # Simple: decay lr
        lr *= 0.999
        
    # Final check
    score = get_score(pts)
    if score > best_score:
        best_score = score
        best_pts = pts.copy()
        
    return best_pts

# EVOLVE-BLOCK-END
