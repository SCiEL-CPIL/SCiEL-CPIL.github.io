import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_radius(points):
    """
    Computes the maximum radius for a set of points.
    r = min(min_{i<j} dist(i,j)/2, min_i (1 - |p_i|))
    """
    if len(points) == 0:
        return 0.0
    
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    # For N=300, 300x300 matrix is 90k elements, very fast
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _objective(flat_points):
    """
    Differentiable objective function to minimize.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist).
    This is equivalent to minimizing the "violation" of a target radius.
    However, since we don't know the optimal r a priori, we can use a 
    "soft" measure of how well the points are spread out and how far they 
    are from the boundary.
    
    A common trick is to minimize the sum of squared overlaps for a fixed 
    radius, but since r varies, we can minimize a function that penalizes 
    small distances.
    
    Let's try minimizing:
    sum_{i<j} max(0, r_target - dist_ij)^2 + sum_i max(0, (1-r_target) - dist_i)^2
    
    But we don't know r_target. 
    Alternative: Minimize the "energy" of a repulsive system.
    E = sum_{i<j} 1/dist_ij^2 + sum_i 1/(1 - dist_i)^2 ?
    This tends to push points apart and to the boundary.
    
    However, L-BFGS-B is gradient-based. The objective must be smooth.
    Let's use the "soft-min" approximation of the packing radius.
    We want to maximize r. 
    Let r_soft = -log(sum(exp(-dist/2/T))) * T ? No, that's for min.
    
    Let's stick to the violation approach but make it adaptive or use a 
    fixed intermediate target? No, that's brittle.
    
    Better approach: Minimize the sum of squared "deficits" relative to a 
    dynamic or fixed target? 
    Actually, a very effective differentiable proxy for packing is:
    Minimize sum_{i<j} (1/d_ij)^k + sum_i (1/(1-d_i))^k
    This repels pairs and pushes to boundary. Minimizing this maximizes the 
    minimum distance in a global sense.
    
    Let's use k=2 for numerical stability (inverse square).
    """
    pts = flat_points.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero on diagonal
    dists = np.sqrt(dists_sq)
    
    # Repulsion term: sum 1/d^2 for i < j
    # We can use the upper triangle
    iu = np.triu_indices(N, k=1)
    repulsion = np.sum(1.0 / (dists[iu[0], iu[1]]**2))
    
    # Boundary term: sum 1/(1-d)^2
    # Clip distances to avoid singularity if point goes outside
    # If point is outside, 1-d is negative. We want to penalize being outside.
    # Let's use (1 - d)^2 in denominator? Or just (1-d) with clipping?
    # If d > 1, 1-d < 0. 1/(1-d)^2 is still positive.
    # But gradient might be unstable.
    # Let's use: sum 1 / max(1e-6, 1 - d)^2
    dists_to_origin = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - dists_to_origin
    # Clip to prevent division by zero or negative issues if we don't square
    # If we square, sign doesn't matter for value, but gradient direction?
    # d/dd (1/(1-d)^2) = 2/(1-d)^3. 
    # If d > 1, 1-d < 0, grad is negative. We want to decrease d. 
    # Minimize -> move in direction of -grad. If grad is neg, -grad is pos. 
    # So it pushes d to increase? Wait.
    # If d=1.1, 1-d=-0.1. Obj = 1/0.01 = 100.
    # If d=1.0, Obj = Inf.
    # If d=0.9, Obj = 1/0.01 = 100.
    # It's symmetric around d=1. It penalizes being close to boundary from inside AND outside.
    # This is good. It keeps points away from the boundary line, but we WANT them close.
    # Wait, we want to MAXIMIZE the radius. 
    # If we minimize repulsion, points spread out. 
    # If we minimize boundary term 1/(1-d)^2, points move AWAY from d=1. 
    # That's wrong. We want them to move TOWARDS d=1.
    
    # So we need a potential that decreases as d approaches 1 from inside.
    # e.g. (1-d)^2. Minimizing (1-d)^2 pushes d to 1.
    # So objective = Repulsion + Boundary_Violation?
    # No, we want to pack them. 
    # Standard approach: Minimize sum of overlaps for a FIXED radius.
    # But we want to find the radius.
    
    # Let's try a different objective: 
    # Minimize the sum of squared "soft overlaps" for a radius r that is 
    # implicitly determined? No.
    
    # Let's go back to: Minimize E = sum_{i<j} 1/d_ij^2.
    # This pushes points apart. 
    # But without boundary constraint, they fly to infinity.
    # We need to keep them inside.
    # So we add a penalty for going outside: sum max(0, d_i - 1)^2.
    # But we also want them to be as close to the boundary as possible to 
    # maximize the minimum distance?
    # Actually, if we just minimize repulsion subject to staying inside, 
    # the points will settle in a configuration that maximizes the minimum 
    # distance? Not necessarily. They might clump in the center to save 
    # "boundary penalty" if we had one, but we don't. 
    # If we just minimize repulsion with a hard boundary constraint (or penalty), 
    # the points will spread out to the boundary to maximize separation.
    
    # Let's use:
    # Objective = sum_{i<j} 1/d_ij^2 + sum_i max(0, d_i - 1 + epsilon)^2
    # The second term prevents them from leaving the circle.
    # The first term pushes them apart.
    # The equilibrium will be a packing.
    
    # To make it differentiable, replace max(0, x) with softplus or just 
    # use (d_i - 1)^2? 
    # If we use (d_i - 1)^2, it pushes them to d=1. 
    # This is perfect. It acts like a "spring" to the boundary.
    
    boundary_penalty = np.sum((dists_to_origin - 1.0)**2)
    
    return repulsion + boundary_penalty

def _gradient(flat_points):
    """
    Analytical gradient of the objective.
    """
    pts = flat_points.reshape(-1, 2)
    
    # Repulsion gradient
    # E_rep = sum_{i<j} d_ij^{-2}
    # dE_rep / dp_i = sum_{j != i} -2 * d_ij^{-3} * (p_i - p_j) / d_ij ? 
    # Let's derive:
    # f(d) = d^{-2}
    # df/dp_i = (df/dd) * (dd/dp_i)
    # dd/dp_i = (p_i - p_j) / d
    # df/dd = -2 d^{-3}
    # So term for pair (i,j) on p_i is: -2 d^{-3} * (p_i - p_j) / d = -2 (p_i - p_j) / d^4
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # We need d^4 = (d^2)^2
    dists_4 = dists_sq**2
    # Avoid div by zero
    dists_4_safe = np.where(dists_4 > 1e-12, dists_4, 1.0)
    
    # Coefficient for each pair (i,j) acting on i is -2 / d^4 * (p_i - p_j)
    # Sum over j
    # Factor matrix M where M[i,j] = -2 / d_ij^4
    M = -2.0 / dists_4_safe
    np.fill_diagonal(M, 0.0)
    
    # grad_rep[i] = sum_j M[i,j] * (p_i - p_j)
    # = sum_j M[i,j] * p_i - sum_j M[i,j] * p_j
    # = p_i * sum_j M[i,j] - sum_j M[i,j] * p_j
    
    sum_M = np.sum(M, axis=1)
    # M @ pts gives sum_j M[i,j] * p_j
    M_pts = M @ pts
    
    grad_rep = pts * sum_M[:, np.newaxis] - M_pts
    
    # Boundary gradient
    # E_bound = sum (d_i - 1)^2
    # dE_bound / p_i = 2 * (d_i - 1) * (p_i / d_i)
    
    dists_to_origin = np.linalg.norm(pts, axis=1)
    dists_to_origin_safe = np.where(dists_to_origin > 1e-12, dists_to_origin, 1.0)
    
    grad_bound = 2.0 * (dists_to_origin - 1.0) / dists_to_origin_safe * pts
    
    return (grad_rep + grad_bound).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_r = -1.0
    
    # Multi-start strategy
    # Try a few random initializations and one hexagonal lattice
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Hexagonal lattice initialization
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            points = None
            while True:
                k = int(1.0 / spacing) + 2
                rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)]
                pts = np.array(rows)
                radius = np.hypot(pts[:, 0], pts[:, 1])
                mask = radius <= 1.0 - spacing / 2.0
                pts = pts[mask]
                if len(pts) >= N:
                    points = pts[np.argsort(radius, kind="stable")[:N]]
                    break
                spacing *= 0.98
        else:
            # Random initialization
            # Use points near the boundary to help convergence?
            # Or random inside?
            # Random inside is safer to avoid local minima near boundary.
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N))
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        current_points = points.copy()
        
        # Optimize
        # L-BFGS-B is good for this.
        # We don't have explicit bounds in the objective, but the penalty keeps them in.
        # We can add bounds [-1, 1] for x and y to help.
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                _objective,
                current_points.flatten(),
                jac=_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < 1e10: # Check for failure
                opt_points = res.x.reshape(-1, 2)
                r = _compute_radius(opt_points)
                
                if r > best_r:
                    best_r = r
                    best_points = opt_points.copy()
                    
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    if best_points is None:
        # Fallback to random if optimization failed completely
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        best_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
    return best_points

