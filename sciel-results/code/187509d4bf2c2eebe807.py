"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _pairwise_min_dist(points):
    """Compute minimum pairwise distance among N points."""
    n = len(points)
    if n < 2:
        return 1.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def _score(points):
    """Compute the packing score: min(min_pairwise/2, min_boundary)."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary = np.min(1.0 - radii)
    # Pairwise constraint
    pd = _pairwise_min_dist(points)
    return min(pd / 2.0, boundary)


def _hexagonal_initial(n):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by radius to prefer interior points
            points = points[np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:n]]
            return points
        spacing *= 0.95


def _random_initial(n):
    """Generate random initial configuration inside the unit circle."""
    while True:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Check minimum distance is reasonable
        if _pairwise_min_dist(points) > 0.05:
            return points


def _objective(points, r_target):
    """
    Objective: maximize the minimum of (pairwise_dist/2 - r_target, boundary - r_target).
    We want to find a configuration where all constraints are >= 0.
    Returns negative of the minimum violation (we minimize this).
    """
    n = len(points)
    # Boundary violations
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_violation = r_target - (1.0 - radii)  # positive if violated
    
    # Pairwise violations
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pairwise_violation = r_target * 2.0 - dists  # positive if violated
    
    # Combine: we want the minimum margin
    # Use a smooth approximation: soft-min
    # Actually, let's just return the minimum violation
    all_violations = np.concatenate([boundary_violation, pairwise_violation[np.triu_indices(n, k=1)]])
    min_violation = np.min(all_violations)
    
    # We want to minimize the maximum violation (i.e., minimize -min_margin)
    # Return negative of the minimum margin (positive means violation)
    return -min_violation


def _refine(points, r_initial, max_iter=5000, tol=1e-10):
    """
    Refine the configuration using gradient-based optimization.
    We maximize the minimum clearance.
    """
    n = len(points)
    
    def objective_and_grad(x):
        pts = x.reshape(n, 2)
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        boundary = 1.0 - radii
        
        # We want to maximize min(min_pairwise/2, min_boundary)
        # This is equivalent to maximizing t subject to:
        #   dist_ij >= 2t for all i<j
        #   |c_i| <= 1-t for all i
        
        # Use a soft-min approach for differentiability
        # f = softmin of (dists/2, boundary)
        
        # Actually, let's use a penalty approach:
        # Minimize the sum of squared violations for a given t
        # But we want to find the best t...
        
        # Alternative: directly optimize to maximize the minimum clearance
        # Let's use a log-sum-exp style soft-min
        
        # Clearances: pairwise/2 and boundary
        pair_clear = dists / 2.0
        # Only consider upper triangle
        iu = np.triu_indices(n, k=1)
        pair_clear_vals = pair_clear[iu]
        bound_clear_vals = boundary
        
        all_clear = np.concatenate([pair_clear_vals, bound_clear_vals])
        
        # Soft-min: -log(sum(exp(-x/beta))) * beta
        beta = 0.01
        # For numerical stability
        m = np.min(all_clear)
        soft_min = m - beta * np.log(np.sum(np.exp(-(all_clear - m) / beta)))
        
        # We want to maximize soft_min, so minimize -soft_min
        obj = -soft_min
        
        # Gradient: d(soft_min)/d(pts)
        # soft_min = m - beta * log(sum(exp(-(all_clear - m)/beta)))
        # d(soft_min)/d(all_clear_k) = -delta_{k,argmin} + (1/beta)*exp(-(all_clear_k-m)/beta)/sum(exp(-(all_clear-m)/beta))
        # For the soft-min, the gradient is a weighted average
        weights = np.exp(-(all_clear - m) / beta)
        weights /= np.sum(weights)
        
        # Gradient of soft_min w.r.t. all_clear
        d_softmin_d_clear = -np.ones_like(all_clear) + weights / beta
        # Wait, let me redo this.
        # soft_min = min_k clear_k approximated by -beta*log(sum(exp(-clear_k/beta)))
        # d(soft_min)/d(clear_k) = exp(-clear_k/beta) / sum_j exp(-clear_j/beta) = weights_k
        # So gradient of -soft_min w.r.t. clear_k is -weights_k
        
        # Now, clear_k for pairwise: dist_ij/2
        # d(dist_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
        # d(dist_ij/2)/d(pts_i) = (pts_i - pts_j)/(2*dist_ij)
        
        # clear_k for boundary: 1 - |c_i|
        # d(1-|c_i|)/d(c_i) = -c_i/|c_i|
        
        grad_pts = np.zeros_like(pts)
        
        # Pairwise contributions
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 1e-10:
                continue
            w = d_softmin_d_clear[idx]  # This is actually -weights for obj
            # d(obj)/d(pts_i) += -w * d(clear)/d(pts_i)
            # d(clear)/d(pts_i) = (pts_i - pts_j)/(2*d)
            grad_pts[i] += -d_softmin_d_clear[idx] * (pts[i] - pts[j]) / (2.0 * d)
            grad_pts[j] += -d_softmin_d_clear[idx] * (pts[j] - pts[i]) / (2.0 * d)
        
        # Boundary contributions
        for idx in range(n):
            r = radii[idx]
            if r < 1e-10:
                continue
            # d(obj)/d(pts_i) += -d_softmin_d_clear[n + idx] * d(clear)/d(pts_i)
            # d(clear)/d(pts_i) = -pts_i/r
            grad_pts[idx] += -d_softmin_d_clear[n + idx] * (-pts[idx] / r)
        
        return obj, grad_pts.flatten()
    
    # Optimize
    x0 = points.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    best_pts = result.x.reshape(n, 2)
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with hexagonal initial
    points = _hexagonal_initial(N)
    best_score = _score(points)
    best_points = points.copy()
    
    # Try to improve via optimization
    # First, estimate a good radius
    r_est = _score(points)
    
    # Iterative refinement: try to increase the radius
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        # Try to refine with current configuration
        refined = _refine(points, r_est, max_iter=2000)
        score = _score(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined.copy()
            points = refined.copy()
            r_est = score
        else:
            # Try a perturbation
            if time.time() > deadline:
                break
            perturb = points + np.random.normal(0, 0.01, points.shape)
            # Keep inside container
            radii = np.sqrt(np.sum(perturb ** 2, axis=1))
            mask = radii > 0.99
            perturb[mask] = points[mask]
            refined2 = _refine(perturb, r_est, max_iter=1000)
            score2 = _score(refined2)
            if score2 > best_score:
                best_score = score2
                best_points = refined2.copy()
                points = refined2.copy()
                r_est = score2
    
    # Final polish
    if time.time() < deadline - 10:
        final = _refine(best_points, best_score, max_iter=5000, tol=1e-12)
        final_score = _score(final)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    return best_points
# EVOLVE-BLOCK-END
