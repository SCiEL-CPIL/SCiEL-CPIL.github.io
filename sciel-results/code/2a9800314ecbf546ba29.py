import numpy as np
import scipy.optimize as so
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Initial Configuration: Hexagonal Lattice
    # Approximate radius for 300 circles in unit disk
    # Area of disk = pi. Area of 300 circles = 300 * pi * r^2.
    # r ~ sqrt(1/300) ~ 0.057.
    # Hex spacing s. Density ~ 0.9069.
    # 300 * s^2 * sqrt(3)/2 ~ pi -> s^2 ~ 2*pi / (300*sqrt(3)) ~ 0.0121 -> s ~ 0.11
    
    def get_hex_lattice():
        points = []
        # Try a range of spacings to get close to N points
        # We want to fill the disk.
        # Let's just generate a large grid and select the best N.
        # Or generate a grid that fits N.
        
        # A better way: Generate hex lattice points, keep those inside disk, 
        # then adjust or select.
        
        # Let's use a simple grid generation that yields ~N points
        # For N=300, roughly sqrt(300) ~ 17.3
        # Let's try to generate a hex lattice with spacing s such that area matches.
        # Area per point ~ s^2 * sqrt(3)/2.
        # N * s^2 * sqrt(3)/2 = pi * 1^2
        # s = sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        # Generate points
        max_r = 1.5
        k = int(max_r / s) + 2
        pts = []
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3.0) / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        
        # Filter points inside the disk (with margin)
        dists = np.linalg.norm(pts, axis=1)
        # We want to keep points that are well inside.
        # Then we might need to select exactly N.
        # If we have more, we can optimize. If fewer, we need to add.
        # Let's just take all points inside 0.95 radius, then add random points if needed?
        # No, let's just use the lattice points closest to center.
        # But we need exactly N.
        
        # Sort by distance to center
        idx = np.argsort(dists)
        pts = pts[idx]
        
        # If we have more than N, take first N?
        # This might leave a hole in the middle if not careful, but hex lattice is dense.
        # Actually, taking the N closest to center is a good heuristic for packing in a disk.
        if len(pts) >= N:
            return pts[:N]
        else:
            # If not enough, add random points
            while len(pts) < N:
                p = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(p) < 0.9:
                    pts = np.vstack([pts, [p]])
            return pts

    x0 = get_hex_lattice()
    
    # Ensure x0 is valid (inside disk)
    # Clip or project? Just ensure it's valid for the optimizer.
    
    best_x = x0.copy()
    best_r = -np.inf
    
    def compute_r(x):
        # x is (N, 2)
        # min_i (1 - |x_i|)
        norms = np.linalg.norm(x, axis=1)
        boundary_min = np.min(1.0 - norms)
        if boundary_min <= 0:
            return 0.0
        # min_{i<j} |x_i - x_j| / 2
        # Compute pairwise distances
        # For N=300, this is 300x300 matrix, fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists) / 2.0
        return min(boundary_min, min_dist)

    # Evaluate initial
    r0 = compute_r(x0)
    best_r = max(best_r, r0)
    best_x = x0.copy()
    
    # Optimization Loop
    # Use soft-min objective
    # f(x) = - soft_min( [1-|x_i|, d_ij/2] )
    
    # We will increase beta
    betas = [50, 100, 200, 400, 800, 1600]
    
    # Precompute indices for efficiency? 
    # For N=300, O(N^2) is fine.
    
    def objective_and_grad(x_flat, beta):
        x = x_flat.reshape(N, 2)
        
        # 1. Boundary terms
        norms = np.linalg.norm(x, axis=1)
        # Avoid division by zero
        norms_safe = np.where(norms < 1e-12, 1e-12, norms)
        boundary_vals = 1.0 - norms
        boundary_grads = -x / norms_safe[:, np.newaxis] # d(1-|x|)/dx = -x/|x|
        
        # 2. Pairwise distances
        # d_ij = |x_i - x_j|
        # We need d_ij / 2
        # Let's compute distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        # Soft min over all values
        # Z = [boundary_vals, pair_vals (flattened, i<j)]
        # To avoid duplicating pair distances, we can just use the full matrix and divide by 2 in the sum?
        # No, soft min is not linear.
        # Let's create a single vector Z.
        # Size: N + N(N-1)/2
        # For N=300, 300 + 44850 = 45150.
        
        # Efficient computation of soft min and gradient
        # Z_k values
        # w_k = exp(-beta * Z_k) / sum(exp(-beta * Z_j))
        # grad_x = sum_k w_k * dZ_k/dx
        
        # Let's compute exp terms
        # boundary_exp = exp(-beta * boundary_vals)
        # pair_exp = exp(-beta * pair_vals)
        
        # Note: pair_vals is symmetric. We should only count each pair once to match the definition of min.
        # min_{i<j} d_ij/2.
        # So Z contains unique pairs.
        
        # Create upper triangle indices
        i_idx, j_idx = np.triu_indices(N, 1)
        pair_vals_unique = pair_vals[i_idx, j_idx]
        
        # Concatenate
        Z = np.concatenate([boundary_vals, pair_vals_unique])
        
        # Compute weights
        # To prevent underflow, subtract min
        min_Z = np.min(Z)
        exp_Z = np.exp(-beta * (Z - min_Z))
        sum_exp = np.sum(exp_Z)
        weights = exp_Z / sum_exp
        
        # Objective value: - soft_min
        # soft_min = min_Z - (1/beta) * ln(sum_exp)
        obj = -(min_Z - (1.0 / beta) * np.log(sum_exp))
        
        # Gradient
        # grad_x = sum_{k in boundary} w_k * grad_b_k + sum_{k in pairs} w_k * grad_p_k
        
        grad = np.zeros_like(x)
        
        # Boundary part
        w_b = weights[:N]
        # grad_b_k = -x_k / |x_k|
        # grad += sum_k w_b_k * (-x_k / |x_k|)
        grad -= np.sum(w_b[:, np.newaxis] * boundary_grads, axis=0)
        
        # Pair part
        w_p = weights[N:]
        # For pair (i,j), value is d_ij/2.
        # d(d_ij/2)/dx_i = (x_i - x_j) / (2 * d_ij) = (x_i - x_j) / (2 * 2 * (d_ij/2)) ?
        # d_ij = |x_i - x_j|.
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij.
        # d(d_ij/2)/dx_i = 0.5 * (x_i - x_j) / d_ij.
        
        # We have weights w_p corresponding to pairs in i_idx, j_idx order.
        # We need to accumulate into grad.
        # grad[i] += w_p * 0.5 * (x_i - x_j) / d_ij
        # grad[j] += w_p * 0.5 * (x_j - x_i) / d_ij
        
        # d_ij for unique pairs
        d_unique = dists[i_idx, j_idx]
        # Avoid div by zero
        d_safe = np.where(d_unique < 1e-12, 1e-12, d_unique)
        vecs = x[i_idx] - x[j_idx]
        unit_vecs = vecs / d_safe[:, np.newaxis]
        
        # Factor 0.5 * w_p
        factors = 0.5 * w_p[:, np.newaxis]
        
        # Accumulate
        # Use np.add.at for scatter
        np.add.at(grad, i_idx, factors * unit_vecs)
        np.add.at(grad, j_idx, -factors * unit_vecs)
        
        return obj, grad.flatten()

    # Optimization
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run optimizer
        # Start from current best_x
        x_flat = best_x.flatten()
        
        # L-BFGS-B
        res = so.minimize(
            lambda xf: objective_and_grad(xf, beta)[0],
            x_flat,
            jac=lambda xf: objective_and_grad(xf, beta)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success:
            x_new = res.x.reshape(N, 2)
            r_new = compute_r(x_new)
            if r_new > best_r:
                best_r = r_new
                best_x = x_new
        
        # Also try a few random perturbations if time allows?
        # No, stick to the plan.

    # Final Polish: Use the exact min objective with a local search?
    # The soft-min with high beta approximates the exact min well.
    # We can do one more step with a very high beta or just return best_x.
    
    # Ensure validity
    # Check if best_x is valid
    r_final = compute_r(best_x)
    if r_final <= 0:
        # Fallback to initial if something went wrong
        return x0
    return best_x
