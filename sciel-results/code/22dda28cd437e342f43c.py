"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def compute_exact_score(points):
        """Compute the true packing radius for a set of points."""
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        r_orig = np.linalg.norm(points, axis=1)
        if np.any(r_orig >= 1.0):
            return 0.0
            
        boundary_margin = 1.0 - r_orig
        min_boundary = np.min(boundary_margin)
        
        # Efficiently find minimum pairwise distance
        # For N=300, O(N^2) is 90k ops, which is fast in NumPy
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist_sq = np.min(dist_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        return min(min_boundary, min_pair_dist / 2.0)

    def soft_min_objective_and_grad(x, beta=50.0):
        """
        Objective: -soft_min(min_pair_dist/2, min_boundary_margin)
        We maximize the radius r, so we minimize -r.
        r = min( min_i (1 - |x_i|), min_{i<j} |x_i - x_j| / 2 )
        
        Using log-sum-exp approximation for the minimum:
        min(a, b) approx -1/beta * log(sum(exp(-beta*a)))
        """
        points = x.reshape(-1, 2)
        
        # 1. Boundary constraints
        r_orig = np.linalg.norm(points, axis=1)
        boundary_margins = 1.0 - r_orig
        
        # 2. Pairwise constraints
        # To save memory and time, we can compute this in chunks or just full matrix for N=300
        # N=300 -> 90k pairs. 300x300x2 array is ~1.4MB. This is fine.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # We want min_{i<j} dist_ij / 2
        # Let v_ij = dist_ij / 2
        # min v_ij approx -1/beta * log( sum exp(-beta * v_ij) )
        
        # Compute all pair distances / 2
        dists = np.sqrt(dist_sq)
        pair_vals = dists / 2.0
        
        # Combine boundary and pair values into one vector to find the global min
        # Actually, we can just take the min of the two soft mins? No, min(min(A), min(B)) = min(A U B)
        # So we can concatenate the vectors.
        
        # To avoid overflow in exp, we use the log-sum-exp trick
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        # min(x) = -LSE(-x)
        
        # Let's form the vector of negative values: -boundary_margins and -pair_vals
        # We want min( boundary_margins, pair_vals )
        # This is equivalent to min( all_elements_in_combined )
        
        # Since N=300, combined vector size is 300 + 44850 = 45150.
        # This is small enough to handle directly.
        
        # However, computing exp for 45k elements every gradient step might be slow if beta is high.
        # Let's stick to a moderate beta.
        
        # For gradient, we need d/dx min( ... )
        # Let m = min(vals)
        # Soft min: S = -1/beta * log( sum exp(-beta * vals) )
        # dS/dval_k = exp(-beta * val_k) / sum exp(-beta * vals)
        # This is a softmax-like distribution over the constraints.
        
        # Let's compute the weights for each constraint.
        
        # Boundary weights
        # val_b_i = 1 - |x_i|
        # Pair weights
        # val_p_ij = |x_i - x_j| / 2
        
        # To be efficient, we can compute the soft min for boundary and pair separately?
        # No, the global min is the min of the union.
        # If we compute soft_min_boundary and soft_min_pair separately, the global soft min is not simply the min of them.
        # But for large beta, it approximates min(min_b, min_p).
        # Actually, min(min_b, min_p) is exactly the min of the union.
        # The soft min of the union is a weighted average.
        
        # Let's compute the combined log-sum-exp.
        # To save memory, we might not want to create the full 300x300 matrix of exp values.
        # But 45k floats is tiny.
        
        # Vector of values:
        # b_vals = boundary_margins (shape N)
        # p_vals = pair_vals (shape N*N, but only upper triangle is unique, symmetric)
        # Note: pair_vals has 0 on diagonal (set to inf).
        
        # Let's just use the full matrix for simplicity, masking diagonal.
        # Actually, we can just use the lower triangle to avoid double counting in the sum?
        # The definition of min doesn't care about duplicates.
        # sum(exp(-beta * vals)) will have duplicates for pairs (i,j) and (j,i).
        # This just doubles the weight of pair constraints relative to boundary.
        # We can correct for this by dividing pair weights by 2 or just accepting the bias.
        # Given N=300, 45k pairs vs 300 boundaries. The pairs dominate the sum.
        # This might make the optimizer focus too much on pairs.
        # Better to use unique pairs.
        
        # Let's get indices for upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        # Extract unique pair distances
        unique_dists = dists[i_idx, j_idx]
        unique_pair_vals = unique_dists / 2.0
        
        # Combined values
        all_vals = np.concatenate([boundary_margins, unique_pair_vals])
        
        # Log-sum-exp for min
        # min(vals) approx -1/beta * log(sum(exp(-beta * vals)))
        
        # For numerical stability:
        # max_val = np.max(-all_vals)  # This is -min(all_vals)
        # log_sum = max_val + log(sum(exp(-all_vals - max_val)))
        # soft_min_val = -1/beta * log_sum
        
        # Gradient:
        # d(soft_min)/d(val_k) = exp(-beta * val_k) / sum(exp(-beta * vals))
        # This is the probability distribution P_k.
        # Then d(soft_min)/dx = sum_k P_k * d(val_k)/dx
        
        neg_vals = -all_vals
        max_neg = np.max(neg_vals)
        shifted = neg_vals - max_neg
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        # Soft min value
        soft_min_val = -1.0 / beta * (max_neg + np.log(sum_exp))
        
        # We want to maximize soft_min_val, so minimize -soft_min_val
        obj = -soft_min_val
        
        # Weights (probabilities)
        weights = exp_shifted / sum_exp
        
        # Gradient of soft_min_val w.r.t x
        grad_x = np.zeros_like(points)
        
        # 1. Boundary part
        # val_b_i = 1 - ||x_i||
        # d(val_b_i)/dx_i = -x_i / ||x_i||
        # weight_b_i = weights[:N]
        
        r_norms = r_orig.copy()
        r_norms[r_norms < 1e-8] = 1.0 # avoid div by zero
        unit_vecs = points / r_norms[:, np.newaxis]
        
        # Contribution to gradient
        # d(soft_min)/dx_i += weight_b_i * (-unit_vec_i)
        grad_x += -weights[:N][:, np.newaxis] * unit_vecs
        
        # 2. Pair part
        # val_p_ij = ||x_i - x_j|| / 2
        # d(val_p_ij)/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||) = (x_i - x_j) / (2 * dist_ij)
        # Note: val_p_ij = dist_ij / 2, so d(val)/dx = (x_i-x_j)/dist_ij * 1/2?
        # Let v = ||u||/2. dv/du = u/(2||u||).
        # Here u = x_i - x_j.
        # So d(val_p_ij)/dx_i = (x_i - x_j) / (2 * dist_ij)
        # And d(val_p_ij)/dx_j = (x_j - x_i) / (2 * dist_ij) = -(x_i - x_j) / (2 * dist_ij)
        
        # We have unique pairs (i, j) with i < j.
        # weight_p_k corresponds to pair (i_k, j_k)
        
        # We need to accumulate gradients for i and j.
        # Using np.add.at for sparse updates
        
        w_pairs = weights[N:]
        
        # Get coordinates
        xi = points[i_idx]
        xj = points[j_idx]
        diff_vecs = xi - xj
        dists_unique = unique_dists
        dists_unique[dists_unique < 1e-8] = 1.0
        
        # Gradient term for each pair
        # grad_i += w_k * diff_vec_k / (2 * dist_k)
        # grad_j -= w_k * diff_vec_k / (2 * dist_k)
        
        coeff = w_pairs / (2.0 * dists_unique)
        
        # Accumulate to grad_x
        # For i:
        np.add.at(grad_x, i_idx, coeff[:, np.newaxis] * diff_vecs)
        # For j:
        np.add.at(grad_x, j_idx, -coeff[:, np.newaxis] * diff_vecs)
        
        # The gradient of the objective (which is -soft_min) is -grad_x
        # Wait, we defined obj = -soft_min_val.
        # d(obj)/dx = - d(soft_min_val)/dx = -grad_x
        
        return obj, -grad_x

    def init_hexagonal(n, seed=42):
        """Initialize n points on a perturbed hexagonal lattice."""
        rng = np.random.RandomState(seed)
        spacing = np.sqrt(np.pi / n) * 0.95
        
        points = []
        max_k = int(2.0 / spacing) + 3
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing * 0.6) ** 2:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > n:
            indices = rng.choice(len(points), size=n, replace=False)
            points = points[indices]
        elif len(points) < n:
            extra = rng.uniform(-0.9, 0.9, size=(n - len(points), 2))
            r = np.hypot(extra[:, 0], extra[:, 1])
            mask = r > 0.85
            while mask.any():
                extra[mask] = rng.uniform(-0.85, 0.85, size=(mask.sum(), 2))
                r = np.hypot(extra[:, 0], extra[:, 1])
                mask = r > 0.85
            points = np.vstack([points, extra])
        
        points = points + rng.normal(0, spacing * 0.05, size=points.shape)
        
        r = np.hypot(points[:, 0], points[:, 1])
        for i in range(len(points)):
            if r[i] > 0.95:
                points[i] *= 0.95 / r[i]
        
        return points

    def init_rings(n, seed=42):
        """Initialize points on concentric rings."""
        rng = np.random.RandomState(seed)
        points = np.zeros((n, 2))
        
        # Approximate number of points per ring
        # Area of ring k is pi * (r_k^2 - r_{k-1}^2)
        # Assume uniform density
        # r_k ~ sqrt(k)
        
        # Simple heuristic:
        # Ring 0: 1 point
        # Ring k: ~ 2 * pi * r_k * dr
        # Let's just distribute randomly within the disk and then sort by radius?
        # Or generate specific rings.
        
        # Let's try to fill rings.
        idx = 0
        r = 0.0
        while idx < n:
            # Estimate how many points fit in next ring
            # This is tricky. Let's just use a fixed number of rings and distribute.
            pass
            
        # Alternative: Random points, then sort by radius and adjust?
        # Let's stick to a simple random disk initialization but with a specific structure.
        # Actually, hexagonal is usually better. Let's add a "grid" init.
        
        # Let's try a polar grid.
        num_rings = int(np.sqrt(n))
        points_list = []
        # Center
        points_list.append([0.0, 0.0])
        idx = 1
        for k in range(1, num_rings + 1):
            r_k = k / num_rings * 0.95
            # Circumference ~ 2 * pi * r_k
            # Spacing ~ sqrt(pi/n)
            count = int(2 * np.pi * r_k / (np.sqrt(np.pi / n) * 0.95))
            if count == 0: count = 1
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.uniform(0, 2*np.pi/count)
            for a in angles:
                if idx >= n: break
                points_list.append([r_k * np.cos(a), r_k * np.sin(a)])
                idx += 1
            if idx >= n: break
            
        points = np.array(points_list[:n])
        # Add noise
        points = points + rng.normal(0, 0.01, size=points.shape)
        
        # Ensure inside
        r = np.hypot(points[:, 0], points[:, 1])
        for i in range(len(points)):
            if r[i] > 0.99:
                points[i] *= 0.99 / r[i]
                
        return points

    def optimize_from_initial(points, beta=50.0):
        x0 = points.flatten()
        
        # Bounds: -1 to 1 for x and y
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # L-BFGS-B is good for large N
        res = minimize(
            soft_min_objective_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False, 'maxls': 50}
        )
        
        return res.x.reshape(-1, 2)

    # Main algorithm
    best_points = None
    best_score = 0.0
    
    initializations = [
        ("hex", lambda s: init_hexagonal(N, seed=s)),
        ("rings", lambda s: init_rings(N, seed=s))
    ]
    
    # Try a few seeds and initialization types
    seeds = [42, 123, 999]
    inits = ["hex", "rings"]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
            
        for init_name in inits:
            if time.time() > deadline - 20.0:
                break
                
            if init_name == "hex":
                points = init_hexagonal(N, seed=seed)
            else:
                points = init_rings(N, seed=seed)
            
            # Optimize
            optimized = optimize_from_initial(points, beta=50.0)
            
            # Compute exact score
            score = compute_exact_score(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
                
            # Try a second pass with higher beta if time permits?
            # Or just refine with a higher beta on the result
            if time.time() < deadline - 10.0:
                refined = optimize_from_initial(best_points, beta=100.0)
                refined_score = compute_exact_score(refined)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined.copy()

    # Final polish: ensure no overlaps and inside circle
    # If best_points is None, use fallback
    if best_points is None:
        best_points = init_hexagonal(N, seed=42)
        
    # Check validity and fix if necessary
    r_orig = np.linalg.norm(best_points, axis=1)
    # Pull any points outside back in
    mask = r_orig > 1.0
    if np.any(mask):
        best_points[mask] *= (0.999 / r_orig[mask])[:, np.newaxis]
        
    # Check for overlaps and push apart?
    # The soft-min optimizer should have handled this, but let's do a quick check.
    # If there are overlaps, the score is 0 or very low.
    # We can do a few steps of repulsion if score is bad.
    
    final_score = compute_exact_score(best_points)
    
    if final_score < 0.01:
        # Fallback to a simple valid packing
        # Grid
        spacing = 0.1
        points = []
        for x in np.arange(-1, 1, spacing):
            for y in np.arange(-1, 1, spacing):
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            # Pad with random
            extra = np.random.uniform(-0.9, 0.9, size=(N - len(points), 2))
            points = np.vstack([points, extra])
        else:
            points = points[:N]
        best_points = points
        
    return best_points
# EVOLVE-BLOCK-END
