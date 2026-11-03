import numpy as np
from scipy.optimize import minimize
import time

N = 50

def soft_min_obj(points, lam):
    """
    Compute log-sum-exp of negative distances and boundary distances.
    Minimizing this maximizes the minimum distance.
    """
    # Reshape points to (N, 2)
    p = points.reshape(-1, 2)
    
    # Pairwise distances
    # Calculate squared distances to avoid sqrt in exp if possible? 
    # exp(-lam * d) is standard.
    # Vectorized pairwise distance calculation
    # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # This is O(N^2). N=50, so 2500 pairs. Fast.
    
    # To avoid self-distance (0), we can mask the diagonal or just sum upper triangle.
    # Let's use upper triangle to save computation and avoid 0 distances.
    idx_i, idx_j = np.triu_indices(N, k=1)
    diff = p[idx_i] - p[idx_j]
    dists = np.linalg.norm(diff, axis=1)
    
    # Boundary distances
    # dist_to_center = np.linalg.norm(p, axis=1)
    # boundary_dists = 1.0 - dist_to_center
    # We want to maximize min(boundary_dists).
    # In the log-sum-exp, we use exp(-lam * boundary_dists).
    # If boundary_dist is negative (outside), exp(-lam * neg) = exp(positive) -> large penalty.
    
    # The objective is log( sum(exp(-lam * dists)) + sum(exp(-lam * boundary_dists)) )
    # Note: The scaling of boundary vs pair distances might need adjustment.
    # Pair distance contributes to radius r via d/2.
    # Boundary distance contributes to radius r via (1-|c|).
    # So they are directly comparable in terms of r.
    # Let's just use the raw distances.
    
    boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
    
    # Combine terms
    # To prevent overflow/underflow in exp, we can use the log-sum-exp trick
    # max_val = np.max(np.concatenate([-lam * dists, -lam * boundary_dists]))
    # But for small N, direct exp is usually fine if lam is not huge.
    # Let's use a stable log-sum-exp implementation.
    
    all_dists = np.concatenate([dists, boundary_dists])
    # We want min(all_dists).
    # log( sum( exp(-lam * d) ) )
    # Let M = max(-lam * d) = -lam * min(d)
    # log( sum( exp(-lam * d) ) ) = M + log( sum( exp(-lam * d - M) ) )
    
    neg_lam_dists = -lam * all_dists
    max_neg = np.max(neg_lam_dists)
    if not np.isfinite(max_neg):
        return 1e10 # Invalid
    
    sum_exp = np.sum(np.exp(neg_lam_dists - max_neg))
    val = max_neg + np.log(sum_exp)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initializations
    # 1. Hexagonal lattice (similar to seed but maybe better tuned)
    # 2. Random
    # 3. Concentric rings
    
    inits = []
    
    # Hexagonal
    # Generate a grid and pick N points closest to center
    spacing = 2.0 / (np.sqrt(N) + 1.5)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius to get the most central N points
            idx = np.argsort(radius, kind="stable")[:N]
            inits.append(points[idx].copy())
            break
        spacing *= 0.95
        
    # Random
    for _ in range(5):
        pts = np.random.uniform(-0.8, 0.8, (N, 2))
        # Scale to fit roughly
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            pts = pts / max_r * 0.9
        inits.append(pts)
        
    # Concentric Rings
    # Try a few ring configurations
    for num_rings in [4, 5, 6]:
        pts = []
        for i in range(num_rings):
            if i == 0:
                pts.append([0, 0])
            else:
                # Distribute points on ring
                # Number of points on ring i?
                # Let's just use random points on a circle of radius i * 0.2
                r = i * 0.18
                angles = np.random.uniform(0, 2*np.pi, N)
                # This is too random. Let's do a structured one.
                pass
        # Skip complex ring init for now, rely on lattice and random + optimization
        
    # Optimization
    # Use L-BFGS-B
    # Objective: soft_min_obj
    # Gradient: Numerical or analytic?
    # Analytic gradient is better.
    
    def obj_and_grad(x):
        p = x.reshape(-1, 2)
        # Compute value
        val = soft_min_obj(p, 15.0)
        
        # Compute gradient
        # d/dx log(sum(exp(-lam * d))) = (1/sum(exp)) * sum(exp * d(exp)/dx)
        # = (1/sum(exp)) * sum(exp * (-lam * d(d)/dx))
        # = -lam * (sum(exp * d(d)/dx)) / sum(exp)
        
        # This requires careful vectorization.
        # Given time constraints, let's use finite differences first to ensure correctness, 
        # then switch to analytic if time permits. 
        # Actually, L-BFGS-B with numerical gradient is slow for 100 vars.
        # Let's implement analytic gradient.
        
        lam = 15.0
        idx_i, idx_j = np.triu_indices(N, k=1)
        diff = p[idx_i] - p[idx_j]
        dists = np.linalg.norm(diff, axis=1)
        # Avoid division by zero if dist is 0
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
        
        all_dists = np.concatenate([dists, boundary_dists])
        neg_lam_dists = -lam * all_dists
        max_neg = np.max(neg_lam_dists)
        exp_terms = np.exp(neg_lam_dists - max_neg)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient of pairwise terms
        # d(dist_ij)/dp_i = (p_i - p_j) / dist_ij
        # d(dist_ij)/dp_j = (p_j - p_i) / dist_ij
        # Term in sum: weights[k] * (-lam) * d(dist)/dp
        # So grad_p_i += weights[k] * (-lam) * (p_i - p_j) / dist_ij
        
        grad = np.zeros_like(p)
        
        # Pairwise
        # diff is p_i - p_j
        # unit_vec = diff / dists[:, np.newaxis]
        unit_vec = diff / dists_safe[:, np.newaxis]
        
        # coeff = weights[0:len(dists)] * (-lam)
        coeff = weights[:len(dists)] * (-lam)
        
        # grad_p_i += coeff[:, np.newaxis] * unit_vec
        # grad_p_j -= coeff[:, np.newaxis] * unit_vec
        
        np.add.at(grad, idx_i, coeff[:, np.newaxis] * unit_vec)
        np.subtract.at(grad, idx_j, coeff[:, np.newaxis] * unit_vec)
        
        # Boundary terms
        # d(boundary_dist_i)/dp_i = d(1 - ||p_i||)/dp_i = - p_i / ||p_i||
        # Term in sum: weights[k] * (-lam) * d(boundary)/dp
        # coeff_b = weights[len(dists):] * (-lam)
        # grad_p_i += coeff_b[:, np.newaxis] * (- p_i / ||p_i||)
        
        norms = np.linalg.norm(p, axis=1)
        norms_safe = np.where(norms < 1e-6, 1e-6, norms)
        unit_p = p / norms_safe[:, np.newaxis]
        
        coeff_b = weights[len(dists):] * (-lam)
        # d(1-||p||)/dp = -unit_p
        # So term is coeff_b * (-unit_p) = -coeff_b * unit_p
        # Wait.
        # Objective: log(sum(exp(-lam * d)))
        # Grad = -lam * sum( w_k * grad(d_k) )
        # For boundary: d_k = 1 - ||p_k||
        # grad(d_k) wrt p_k = - p_k / ||p_k|| = -unit_p_k
        # So contribution is -lam * w_k * (-unit_p_k) = lam * w_k * unit_p_k
        # My coeff_b is weights * (-lam).
        # So contribution is coeff_b * (-unit_p) ? No.
        # Let's re-derive.
        # f = log( sum exp(-lam d) )
        # df/dx = (1/sum) sum exp(-lam d) * (-lam * dd/dx)
        # = -lam * sum ( w * dd/dx )
        # For boundary: dd/dx = -unit_p
        # df/dx = -lam * w * (-unit_p) = lam * w * unit_p
        # My coeff_b = w * (-lam).
        # So df/dx = -coeff_b * unit_p
        grad += -coeff_b[:, np.newaxis] * unit_p
        
        return val, grad

    # Try each initialization
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                obj_and_grad,
                init.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < best_score:
                # Calculate actual score (raw radius)
                p = res.x.reshape(-1, 2)
                # Check validity
                if np.any(np.linalg.norm(p, axis=1) > 1.0):
                    # If outside, shrink? Or just reject?
                    # The soft penalty should keep them in.
                    pass
                
                # Calculate true min radius
                idx_i, idx_j = np.triu_indices(N, k=1)
                diff = p[idx_i] - p[idx_j]
                dists = np.linalg.norm(diff, axis=1)
                min_pair_dist = np.min(dists)
                
                boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
                min_boundary_dist = np.min(boundary_dists)
                
                r = min(min_pair_dist / 2.0, min_boundary_dist)
                
                if r > best_score:
                    best_score = r
                    best_points = p
                    
        except Exception as e:
            pass

    # If no valid points found, return seed-like
    if best_points is None:
        # Fallback to the first init
        best_points = inits[0]
        
    return best_points
