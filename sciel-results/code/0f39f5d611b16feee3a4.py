import numpy as np
import time
import scipy.optimize
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy:
    1. Multi-start optimization using a smooth surrogate objective.
       The objective is the log-sum-exp (soft-max) of the squared inverse distances,
       which approximates the log of the maximum pairwise distance.
       We minimize this surrogate, which is equivalent to maximizing the minimum distance
       (in a normalized sense where we push the max distance down).
       
       Actually, a better smooth surrogate for "maximize min distance subject to max distance <= 1" 
       is to minimize the soft-max of pairwise distances while penalizing violations of min distance.
       
       Let's use the following formulation:
       Minimize: soft_max(||p_i - p_j||^2) - beta * min(||p_i - p_j||^2)
       Wait, min is non-smooth. 
       
       Alternative:
       Maximize: min_{i<j} ||p_i - p_j||^2 subject to max_{i<j} ||p_i - p_j||^2 <= 1
       
       Smooth approximation:
       Let D_max_smooth = (1/kappa) * log(sum(exp(kappa * ||p_i - p_j||^2)))
       Let D_min_smooth = -(1/kappa) * log(sum(exp(-kappa * ||p_i - p_j||^2)))
       
       We want to maximize D_min_smooth / D_max_smooth.
       Or simply maximize D_min_smooth - lambda * D_max_smooth.
       
       Let's try minimizing: D_max_smooth - D_min_smooth.
       This encourages all distances to be equal and as large as possible?
       No, if we just minimize D_max - D_min, we can shrink all points to 0 (D_max=0, D_min=0, diff=0).
       
       We need a normalization.
       Standard approach: Fix the centroid at 0 and scale such that sum ||p_i||^2 = 1 (or similar).
       Then maximize the minimum distance.
       
       Let's use the constraint: sum_{i<j} ||p_i - p_j||^2 = 1 (fixed diameter/size).
       Then maximize min ||p_i - p_j||.
       
       Smooth objective to minimize:
       f(P) = -(1/kappa) * log(sum_{i<j} exp(-kappa * ||p_i - p_j||^2)) + lambda * (sum_{i<j} ||p_i - p_j||^2 - 1)^2
       
       This maximizes the soft-min distance while keeping the total energy fixed.
       
    2. Final polish using exact SLSQP on the true objective:
       Maximize min_dist
       Subject to max_dist <= 1.0 (or scale at the end)
       
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    pair_indices = np.triu_indices(n, k=1)
    num_pairs = len(pair_indices[0])
    i_idx, j_idx = pair_indices

    def get_pairwise_distances(points):
        return np.linalg.norm(points[i_idx] - points[j_idx], axis=1)

    def score(points):
        dists = get_pairwise_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_obj(x, kappa, lambda_energy):
        """
        Minimize: -soft_min(dist^2) + lambda * (sum(dist^2) - 1)^2
        soft_min(z) = -1/kappa * log(sum(exp(-kappa * z)))
        """
        p = x.reshape(n, d)
        # Center to remove translation invariance
        p = p - np.mean(p, axis=0)
        
        diffs = p[i_idx] - p[j_idx]
        dists_sq = np.sum(diffs**2, axis=1)
        
        # Soft min of dists_sq
        # We want to maximize this, so we minimize the negative.
        # To avoid overflow/underflow in exp, subtract max.
        # For soft min: -1/kappa * log(sum(exp(-kappa * dists_sq)))
        # Let m = min(dists_sq). exp(-kappa * (dists_sq - m)) is bounded.
        m = np.min(dists_sq)
        if m < 1e-12:
            m = 1e-12
        exp_terms = np.exp(-kappa * (dists_sq - m))
        soft_min_val = m - (1.0 / kappa) * np.log(np.sum(exp_terms))
        
        # Energy constraint: sum(dists_sq) should be close to 1
        energy = np.sum(dists_sq)
        energy_penalty = lambda_energy * (energy - 1.0)**2
        
        return -soft_min_val + energy_penalty

    def smooth_grad(x, kappa, lambda_energy):
        p = x.reshape(n, d)
        p = p - np.mean(p, axis=0)
        
        diffs = p[i_idx] - p[j_idx]
        dists_sq = np.sum(diffs**2, axis=1)
        
        m = np.min(dists_sq)
        if m < 1e-12:
            m = 1e-12
        
        exp_terms = np.exp(-kappa * (dists_sq - m))
        sum_exp = np.sum(exp_terms)
        
        # Gradient of soft_min:
        # d/dz_j soft_min = exp(-kappa(z_j - m)) / sum_exp
        # Note: m depends on z, but for large kappa, the term with min z dominates.
        # Actually, derivative of -1/kappa log(sum exp(-kappa(z_j - m))) 
        # Let's just use numerical gradient or a simpler analytical form.
        # For large kappa, soft_min approx min. The gradient is 0 for non-min terms and 1 for min terms (averaged).
        # The analytical gradient of g(z) = -1/kappa log(sum exp(-kappa z_j)) is:
        # g'(z_j) = exp(-kappa z_j) / sum(exp(-kappa z_k))
        
        weights = exp_terms / sum_exp
        
        # Gradient of dists_sq wrt p:
        # d/dp_i ||p_i - p_j||^2 = 2(p_i - p_j)
        # d/dp_j ||p_i - p_j||^2 = -2(p_i - p_j)
        
        grad_p = np.zeros_like(p)
        
        # Accumulate gradients for soft_min
        # d/dp soft_min = sum_j weights_j * d/dp dists_sq_j
        for idx in range(num_pairs):
            i, j = i_idx[idx], j_idx[idx]
            w = weights[idx]
            diff = diffs[idx]
            grad_p[i] += 2 * w * diff
            grad_p[j] -= 2 * w * diff
            
        # Gradient of energy penalty:
        # E = sum dists_sq
        # dE/dp_i = 2 * sum_{j!=i} (p_i - p_j) = 2 * (deg_i * p_i - sum_{j!=i} p_j)
        # Simpler: just sum over pairs
        grad_energy = np.zeros_like(p)
        for idx in range(num_pairs):
            i, j = i_idx[idx], j_idx[idx]
            diff = diffs[idx]
            grad_energy[i] += 2 * diff
            grad_energy[j] -= 2 * diff
            
        # Total gradient
        # Objective is -soft_min + lambda * (E - 1)^2
        # d/dp = - grad_soft_min + lambda * 2(E-1) * grad_energy
        
        energy = np.sum(dists_sq)
        grad_obj = -grad_p + lambda_energy * 2 * (energy - 1.0) * grad_energy
        
        return grad_obj.flatten()

    def exact_obj(x):
        """
        Minimize -min_dist subject to max_dist <= 1.
        SLSQP handles constraints.
        """
        p = x.reshape(n, d)
        dists = get_pairwise_distances(p)
        d_min = np.min(dists)
        d_max = np.max(dists)
        # We want to maximize d_min / d_max.
        # If we fix d_max <= 1, we maximize d_min.
        # Return -d_min
        return -d_min

    def exact_constr(x):
        """
        Constraint: max_dist - 1.0 <= 0
        """
        p = x.reshape(n, d)
        dists = get_pairwise_distances(p)
        return 1.0 - np.max(dists)

    # Initial candidates
    candidates = []
    
    # 1. Hexagonal lattice
    x_coords = []
    y_coords = []
    spacing = 1.0
    for row in range(4):
        for col in range(4):
            x = col * spacing
            if row % 2 == 1:
                x += 0.5 * spacing
            y = row * spacing * np.sqrt(3) / 2
            x_coords.append(x)
            y_coords.append(y)
    hex_pts = np.column_stack((x_coords, y_coords))
    center = np.mean(hex_pts, axis=0)
    hex_pts -= center
    candidates.append(hex_pts)
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # 3. Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    rand_pts -= np.mean(rand_pts, axis=0)
    candidates.append(rand_pts)

    best_points = candidates[0]
    best_score = score(best_points)

    # Phase 1: Smooth Optimization
    # Use a moderate kappa to balance smoothness and accuracy
    kappa = 50.0
    lambda_energy = 100.0
    
    for init_pts in candidates:
        if time.time() > deadline - 120:
            break
            
        # Normalize initial points to have sum dist^2 = 1
        dists_init = get_pairwise_distances(init_pts)
        energy_init = np.sum(dists_init**2)
        if energy_init > 0:
            init_pts = init_pts / np.sqrt(energy_init)
        else:
            init_pts = np.random.randn(n, d) / np.sqrt(n)
            
        x0 = init_pts.flatten()
        
        try:
            # Use L-BFGS-B with numerical gradient if analytical is tricky, 
            # but let's try L-BFGS-B with the provided smooth_grad.
            # L-BFGS-B is efficient for large n.
            res = scipy.optimize.minimize(
                smooth_obj,
                x0,
                method='L-BFGS-B',
                jac=smooth_grad,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            pts_smooth = res.x.reshape(n, d)
            # Normalize
            dists_s = get_pairwise_distances(pts_smooth)
            e_s = np.sum(dists_s**2)
            if e_s > 0:
                pts_smooth = pts_smooth / np.sqrt(e_s)
            
            s = score(pts_smooth)
            if s > best_score:
                best_score = s
                best_points = pts_smooth.copy()
                
        except Exception:
            pass

    # Phase 2: Exact Local Polish
    # Use SLSQP to maximize min_dist subject to max_dist <= 1
    # Start from best_points found in Phase 1
    
    if time.time() < deadline - 60:
        # Normalize best_points so that max_dist is close to 1
        dists_b = get_pairwise_distances(best_points)
        d_max_b = np.max(dists_b)
        if d_max_b > 0:
            best_points = best_points / d_max_b
            
        x0 = best_points.flatten()
        
        # Bounds: Points should be within [-1, 1] roughly if max_dist <= 1
        bounds = [(-1.5, 1.5)] * (n * d)
        
        try:
            res = scipy.optimize.minimize(
                exact_obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': exact_constr}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                pts_polished = res.x.reshape(n, d)
                # Verify constraint
                dists_p = get_pairwise_distances(pts_polished)
                d_max_p = np.max(dists_p)
                if d_max_p <= 1.0 + 1e-8:
                    # Normalize to exactly max_dist = 1 for consistency, though ratio is scale invariant
                    if d_max_p > 0:
                        pts_polished = pts_polished / d_max_p
                    
                    s = score(pts_polished)
                    if s > best_score:
                        best_score = s
                        best_points = pts_polished.copy()
                        
        except Exception:
            pass

    # Phase 3: Final High-Precision Polish
    # Run SLSQP again with tighter tolerance if time permits
    if time.time() < deadline - 30:
        dists_b = get_pairwise_distances(best_points)
        d_max_b = np.max(dists_b)
        if d_max_b > 0:
            best_points = best_points / d_max_b
            
        x0 = best_points.flatten()
        bounds = [(-1.5, 1.5)] * (n * d)
        
        try:
            res = scipy.optimize.minimize(
                exact_obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': exact_constr}],
                options={'maxiter': 1000, 'ftol': 1e-15, 'maxls': 200}
            )
            
            if res.success:
                pts_final = res.x.reshape(n, d)
                dists_f = get_pairwise_distances(pts_final)
                d_max_f = np.max(dists_f)
                if d_max_f <= 1.0 + 1e-8:
                    if d_max_f > 0:
                        pts_final = pts_final / d_max_f
                    s = score(pts_final)
                    if s > best_score:
                        best_score = s
                        best_points = pts_final.copy()
                        
        except Exception:
            pass

    return best_points.reshape(16, 2)

if __name__ == "__main__":
    pts = min_max_dist_dim2_16()
    print(pts.shape)
