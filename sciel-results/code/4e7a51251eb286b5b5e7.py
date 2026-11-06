import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Fibonacci Spiral in 2D
    # This provides a uniform distribution of points in the unit disk.
    n = N
    i = np.arange(1, n + 1)
    # Golden angle
    phi = np.pi * (3. - np.sqrt(5.))
    # Radius grows with sqrt(i) to ensure uniform area density
    r = np.sqrt(i / n)
    theta = i * phi
    
    # Start slightly inside to avoid boundary issues initially
    points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # Evaluate actual radius
    def get_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    best_points = points.copy()
    best_radius = get_radius(points)
    
    # 2. Optimization using Soft-Min Surrogate
    # We maximize the soft-min of the set of constraints:
    # { d_ij / 2 for i < j } U { 1 - |c_i| for i }
    
    # We will iterate with increasing beta (sharpness of the soft-min)
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Number of steps per beta stage
        # Reduce steps as beta increases (finer tuning)
        steps = 2000
        lr = 0.01 / beta # Learning rate scales with beta? Or fixed?
        # Let's use a fixed small step size and let the gradient magnitude guide it.
        # Actually, gradient of softmin is bounded.
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
            
            # Compute pairwise differences and distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # We need gradients of dists w.r.t points.
            # grad d_ij / d c_k = (c_k - c_j) / d_ij if k==i, (c_k - c_i) / d_ij if k==j, 0 otherwise.
            # This is tricky to vectorize for the weighted sum directly.
            
            # Alternative: Compute the gradient of the objective function directly.
            # Objective: f = -1/beta * ln( sum_{pairs} exp(-beta * d_ij/2) + sum_{i} exp(-beta * (1-|c_i|)) )
            # Note: The task defines r = min( min d_ij/2, min (1-|c_i|) ).
            # So the terms in the sum should be the values that determine r.
            # Let's define x_k as the "slacks".
            # x_{ij} = d_ij / 2
            # x_{i,b} = 1 - |c_i|
            
            # To compute gradient efficiently:
            # We need weights w_k = exp(-beta * x_k).
            
            # 1. Pairwise terms
            # Mask upper triangle to avoid double counting? 
            # The sum in log-sum-exp doesn't care about order, but if we sum all i!=j, we double count.
            # It's better to sum over i<j.
            # However, vectorizing the gradient contribution from i<j to points is a bit complex.
            # Let's use the full matrix but divide by 2? No, the gradient of d_ij is symmetric.
            # If we sum over all i!=j, the gradient for point i gets contributions from all j.
            # Let's just sum over all i < j.
            
            # Create a mask for i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Get the distances for i < j
            # We can't just index dists with mask easily to get a flat array and then map back.
            # Let's compute the weights for all pairs i!=j, then we can use the full symmetry.
            # If we sum over all i!=j, the objective is -1/beta ln( sum_{i!=j} ... ).
            # This is equivalent to -1/beta ln( (sum_{i<j} ...)^2 ) = -2/beta ln(sum_{i<j} ...).
            # The gradient direction is the same, just scaled.
            # So we can sum over all i!=j for simplicity.
            
            # Set diagonal to inf so exp(-beta*inf) = 0
            dists_diag = dists.copy()
            np.fill_diagonal(dists_diag, np.inf)
            
            # x_pair = dists_diag / 2
            x_pair = dists_diag / 2.0
            w_pair = np.exp(-beta * x_pair)
            
            # Boundary terms
            norms = np.linalg.norm(points, axis=1)
            x_bound = 1.0 - norms
            w_bound = np.exp(-beta * x_bound)
            
            # Total sum in the log
            # S = sum(w_pair) + sum(w_bound)
            S = np.sum(w_pair) + np.sum(w_bound)
            
            # If S is 0 or very small, gradient is undefined or zero. 
            # If S is huge, we might have numerical issues.
            
            # Gradient of f w.r.t points:
            # f = -1/beta ln(S)
            # df/dc_k = -1/beta * (1/S) * dS/dc_k
            # dS/dc_k = sum_{j!=k} w_{kj} * d(d_kj/2)/dc_k + w_{k,b} * d(1-|c_k|)/dc_k
            
            # d(d_kj/2)/dc_k = (c_k - c_j) / (2 * d_kj) * 2? No.
            # d_kj = |c_k - c_j|.
            # d(d_kj)/dc_k = (c_k - c_j) / d_kj.
            # So d(d_kj/2)/dc_k = (c_k - c_j) / (2 * d_kj).
            
            # Let's compute the vector sum of (c_k - c_j) / d_kj weighted by w_kj.
            # Let V_k = sum_{j!=k} w_{kj} * (c_k - c_j) / d_kj.
            
            # We can compute this using matrix operations.
            # W is the weight matrix (N x N).
            # We want sum_j W[k,j] * (P[k] - P[j]) / D[k,j].
            
            # To avoid division by zero on diagonal (which is 0 weight anyway), 
            # we can set D to 1 where it is 0.
            D_safe = np.where(dists_diag == 0, 1.0, dists_diag)
            
            # Unit vectors U[k,j] = (P[k] - P[j]) / D[k,j]
            # P[:, np.newaxis, :] - P[np.newaxis, :, :] is the difference vector.
            diff_vec = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            unit_vec = diff_vec / D_safe[:, :, np.newaxis]
            
            # Zero out diagonal unit vectors (should be 0/1=0 anyway, but safe)
            np.fill_diagonal(unit_vec[:, :, 0], 0.0)
            np.fill_diagonal(unit_vec[:, :, 1], 0.0)
            
            # Weighted sum: V = W @ unit_vec? 
            # V[k, d] = sum_j W[k, j] * unit_vec[k, j, d]
            # Yes, matrix multiplication works.
            V = np.sum(w_pair[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Gradient from pairs: grad_pair[k] = -1/beta * (1/S) * V[k] * 0.5?
            # Wait, x_pair = d/2.
            # d(x_pair)/dc = 0.5 * d(d)/dc = 0.5 * unit_vec.
            # So dS_pair/dc_k = sum_j w_kj * 0.5 * unit_vec_kj.
            # So V should be multiplied by 0.5.
            V = 0.5 * V
            
            # Gradient from boundary:
            # x_bound = 1 - |c_k|.
            # d(x_bound)/dc_k = - c_k / |c_k| = - norm_vec_k.
            # norm_vec = points / norms[:, np.newaxis]
            # Handle norms=0
            norms_safe = np.where(norms == 0, 1.0, norms)
            norm_vec = points / norms_safe[:, np.newaxis]
            
            # dS_bound/dc_k = w_bound[k] * (-norm_vec[k])
            grad_bound = -w_bound[:, np.newaxis] * norm_vec
            
            # Total Gradient
            grad_total = (V + grad_bound) / (beta * S)
            
            # Gradient Ascent
            # Move in direction of grad_total
            # We need a learning rate.
            # The magnitude of grad_total depends on beta and S.
            # A safe step size is needed.
            # Let's use a fixed step size scaled by 1/beta?
            # Or just a small constant.
            step_size = 0.05
            # Normalize gradient? No, magnitude is important.
            # But if S is very small (large distances), gradient is small.
            # If S is large (small distances), gradient is large.
            # This is self-regulating to some extent.
            
            points = points + step_size * grad_total
            
            # Project back to unit disk? 
            # The soft-min penalty should keep them in, but just in case:
            # If any point is outside, pull it in.
            # Actually, if a point goes outside, x_bound becomes negative, 
            # exp(-beta * negative) = exp(positive) -> Huge.
            # This will create a massive gradient pushing it back.
            # So no hard projection needed, but maybe a soft clamp helps stability?
            # Let's stick to pure gradient for now.
            
            # Update best
            # Check every 100 steps
            if step % 100 == 0:
                current_radius = get_radius(points)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    # Final Polish: Local search on the best points found
    # Use the remaining time to do a few iterations of "push apart closest pairs"
    if time.time() < deadline - 5.0:
        points = best_points.copy()
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            # Find closest pair
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            
            # Check boundary
            norms = np.linalg.norm(points, axis=1)
            min_bound_idx = np.argmin(1.0 - norms)
            
            # Decide which constraint is tighter
            d_min = dists[i, j] / 2.0
            b_min = 1.0 - norms[min_bound_idx]
            
            # Push apart
            # If pair is tighter, push i and j apart
            # If boundary is tighter, push point in
            
            # Simple heuristic:
            # 1. Push closest pair apart
            vec = points[i] - points[j]
            d = np.linalg.norm(vec)
            if d > 1e-6:
                # Move each by epsilon/2
                eps = 1e-4
                points[i] += (vec / d) * eps
                points[j] -= (vec / d) * eps
            
            # 2. Push boundary violators in
            for idx in np.where(norms > 1.0)[0]:
                vec = points[idx]
                d = np.linalg.norm(vec)
                if d > 1e-6:
                    points[idx] *= (1.0 - 1e-4) / d # Scale down to be slightly inside
            
            # Also push points that are close to boundary but inside?
            # The gradient method handled this. This is just a fixer.
            
            current_radius = get_radius(points)
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
    return best_points
