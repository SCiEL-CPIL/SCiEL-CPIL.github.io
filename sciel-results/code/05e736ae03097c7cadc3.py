"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def generate_fibonacci_disk(n):
        """
        Generate n points in the unit disk using a Fibonacci-like distribution.
        This provides a more uniform, non-lattice starting configuration.
        """
        points = []
        # Use a spiral pattern
        # r = sqrt(i / n) for uniform area density
        # theta = i * golden_angle
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = np.sqrt(i / n)
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Ensure inside unit circle (r <= 1)
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
        
        # If we need exactly N points, and we have fewer, add more or adjust
        # But Fibonacci spiral naturally gives n points.
        # For n=300, this should be fine.
        return np.array(points[:N])

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # 1. Initialization: Fibonacci Disk
    points = generate_fibonacci_disk(N)
    
    # Ensure we have exactly N points and they are valid
    if len(points) < N:
        # Fallback to random if something went wrong
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.9
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Project to unit disk if necessary (Fibonacci should be safe, but just in case)
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / radii[outside, np.newaxis]

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-Based Relaxation with Soft-Min Surrogate
    # We want to maximize r = min(min_pair_dist/2, min_boundary_dist)
    # Let f_ij = dist(i,j)/2, g_i = 1 - dist(i, origin)
    # We use a soft-min: S = -1/beta * log(sum(exp(-beta * x_k)))
    # where x_k are all f_ij and g_i.
    # Gradient of S w.r.t. position p_i:
    # dS/dp_i = -1/beta * sum_k (d x_k / d p_i) * (exp(-beta * x_k) / sum_j exp(-beta * x_j))
    # Let w_k = exp(-beta * x_k) / sum_j exp(-beta * x_j)
    # For pair (i,j): x_ij = dist(i,j)/2
    # d x_ij / d p_i = (p_i - p_j) / (2 * dist(i,j))
    # So force contribution from pair (i,j) on i is: -1/beta * w_ij * (p_i - p_j) / (2 * dist(i,j))
    # Wait, the sign: we want to maximize S.
    # Gradient ascent: p_i += alpha * dS/dp_i
    # dS/dp_i = sum_k w_k * (d x_k / d p_i) ... No, derivative of -log(sum exp(-beta x)) is 
    # -1 * (1/sum exp) * sum (exp(-beta x_k) * (-beta * dx_k)) = beta * sum (w_k * dx_k)
    # So dS/dp_i = beta * sum_k w_k * (d x_k / d p_i)
    
    # Parameters
    beta = 200.0  # Higher beta makes soft-min closer to hard-min
    step_size = 0.01
    decay_rate = 0.9999
    
    # Precompute indices for efficiency? No, N=300 is small enough for dense matrices.
    
    for it in range(10000):
        if time.time() > deadline:
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Compute soft-min weights
        # x values: all pairwise dists/2 and all boundary dists
        # To avoid numerical overflow in exp(-beta * x), we can shift by min_x
        # But beta * x might be large. Let's use log-sum-exp trick.
        
        # Let's compute weights directly.
        # w_ij = exp(-beta * dists_ij/2) / Z
        # w_i  = exp(-beta * boundary_i) / Z
        
        # Z = sum_ij exp(-beta * dists_ij/2) + sum_i exp(-beta * boundary_i)
        
        # For numerical stability, let m = min(all x)
        # Then exp(-beta * x) = exp(-beta * (x - m)) * exp(-beta * m)
        # The exp(-beta * m) cancels in the ratio.
        
        # x_pair = dists / 2
        x_pair = dists / 2.0
        x_bound = boundary_dists
        
        # Combine into a single vector for min calculation
        # We can't easily combine without flattening, but we can find min separately.
        min_x_pair = np.min(x_pair)
        min_x_bound = np.min(x_bound)
        min_x = min(min_x_pair, min_x_bound)
        
        # Compute unnormalized weights
        # w_pair_ij = exp(-beta * (x_pair_ij - min_x))
        # w_bound_i = exp(-beta * (x_bound_i - min_x))
        
        # Be careful with inf on diagonal of dists
        # x_pair has inf on diagonal. exp(-beta * inf) = 0.
        
        # Use np.exp with clipping to avoid underflow?
        # If beta * (x - min_x) is large, exp is 0. That's fine.
        
        # Calculate weights
        w_pair = np.exp(-beta * (x_pair - min_x))
        np.fill_diagonal(w_pair, 0.0) # Diagonal is 0 anyway
        
        w_bound = np.exp(-beta * (x_bound - min_x))
        
        Z = np.sum(w_pair) + np.sum(w_bound)
        
        if Z == 0:
            break # Degenerate case
        
        w_pair /= Z
        w_bound /= Z
        
        # Calculate forces
        # Force on i from pair (i,j):
        # F_pair_i = beta * sum_j w_pair_ij * (d x_ij / d p_i)
        # d x_ij / d p_i = (p_i - p_j) / (2 * dists_ij)
        # Note: diff[i,j] = p_i - p_j
        # So term is w_pair[i,j] * diff[i,j] / (2 * dists[i,j])
        
        # We need to sum over j for each i.
        # factor_ij = w_pair[i,j] / (2 * dists[i,j])
        # F_pair_i = beta * sum_j factor_ij * diff[i,j]
        
        # Handle dists = 0 (should not happen with inf diagonal, but if points coincide)
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        factor = w_pair / (2.0 * safe_dists)
        np.fill_diagonal(factor, 0.0)
        
        # F_pair_i = sum_j factor[i,j] * (p_i - p_j)
        # This is a matrix multiplication: factor @ diff?
        # factor is (N, N), diff is (N, N, 2)
        # We want sum_j factor[i,j] * diff[i,j, :]
        # This is not a standard matmul.
        # Let's do it component-wise or use einsum.
        # F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        # Force on i from boundary:
        # F_bound_i = beta * w_bound_i * (d x_bound_i / d p_i)
        # x_bound_i = 1 - |p_i|
        # d x_bound_i / d p_i = -p_i / |p_i|
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        # Total force
        forces = beta * (F_pair + F_bound)
        
        # Update positions
        points += forces * step_size
        
        # Project to unit disk
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        # Decay step size
        step_size *= decay_rate
        
        # Occasional perturbation to escape local minima
        if it % 100 == 0 and it > 0:
            num_perturb = N // 20
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * 0.01
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]

    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
