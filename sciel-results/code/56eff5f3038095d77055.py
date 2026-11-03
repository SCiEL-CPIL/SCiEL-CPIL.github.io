import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        valid: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        # Ensure normal points outward
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        # For a convex hull of points on the sphere, the circumcenter is the
        # point on the sphere furthest from the facet plane in the direction of the normal.
        # The angular distance from the center to any vertex is the circumradius.
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros_like(points)
    
    # Use only valid facets
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (r_valid - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    for k in range(len(f_valid)):
        if weights[k] < 1e-10:
            continue
        
        tri = f_valid[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        # Gradient of r = arccos(c . p) w.r.t. p is -(c - (c.p)p) / sin(r)
        # We want to decrease r, so move p in direction of -grad = (c - (c.p)p) / sin(r)
        # But we are doing gradient descent on the objective, so we subtract grad.
        # grad_obj = sum w_k * grad_r_k
        # grad_r_k = - (c - (c.p_k)p_k) / sin(r_k)
        # So -grad_obj = sum w_k * (c - (c.p_k)p_k) / sin(r_k)
        # We want to move points to decrease objective, so step = -lr * grad_obj
        # Which is equivalent to step = lr * sum w_k * (c - (c.p_k)p_k) / sin(r_k)
        
        for idx, p in zip(tri, [p1, p2, p3]):
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Direction to move p to decrease r: towards c (tangent component)
                direction = c - dot_cp * p
                # The gradient magnitude is 1/sin(r).
                # So the step contribution is w_k * (c - dot_cp * p) / sin_r^2 ?
                # Let's just use the analytic gradient of arccos(x) = -1/sqrt(1-x^2)
                # d(arccos(c.p))/dp = -(c - (c.P)P) / sqrt(1-(c.P)^2)
                # So grad_r = - (c - dot_cp * p) / sin_r
                # We want to minimize, so we move in direction -grad_r = (c - dot_cp * p) / sin_r
                # Wait, the code below adds to grad, and later does points - lr * grad.
                # So we need grad = d(objective)/dp.
                # d(objective)/dp = sum w_k * d(r_k)/dp
                # d(r_k)/dp = - (c - dot_cp * p) / sin_r
                # So grad += w_k * (-(c - dot_cp * p) / sin_r)
                # Then step = -lr * grad = lr * w_k * (c - dot_cp * p) / sin_r
                # This moves p towards c. Correct.
                
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane (should already be, but just in case)
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += weights[k] * grad_p
    
    return float(log_sum_exp), grad

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _lloyd_iteration(points):
    """
    Perform one iteration of Lloyd's algorithm.
    1. Compute convex hull.
    2. For each point, find the circumcenters of the triangles incident to it.
    3. Move the point to the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Initialize sum of circumcenters for each point
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        radii, centers, valid = _facet_circumradii(points, facets)
        
        for idx in range(len(facets)):
            if not valid[idx]:
                continue
            i, j, k = facets[idx]
            c = centers[idx]
            
            # This circumcenter c is a vertex of the Voronoi cell for i, j, and k
            sums[i] += c
            sums[j] += c
            sums[k] += c
            
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        
        new_points = points.copy()
        for i in range(N):
            if counts[i] > 0:
                # Move to the normalized sum of the circumcenters
                # This approximates the centroid of the Voronoi cell
                new_points[i] = sums[i]
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] /= norm
                else:
                    # Fallback if sum is zero (should not happen for well-distributed points)
                    new_points[i] = points[i]
            
        return new_points
    except Exception:
        return points

def _gradient_descent(points, start_time, deadline, initial_lr=0.005, min_lr=1e-6, initial_beta=30.0):
    """
    Optimize using gradient descent on a smooth surrogate objective.
    Uses log-sum-exp for soft-max of circumradii with adaptive temperature.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _covering_radius_rad(current_points)
    lr = initial_lr
    beta = initial_beta
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline:
        obj_val, grad = _soft_max_objective_and_grad(current_points, beta)
        
        # Project gradient onto tangent space (remove radial component)
        radial = np.einsum('ij,ij->i', grad, current_points)
        grad -= radial[:, None] * current_points
        
        # Take gradient step
        new_points = current_points - lr * grad
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius - 1e-12:
            best_radius = current_radius
            best_points = new_points.copy()
            current_points = new_points.copy()
            stall_count = 0
            # Decay learning rate slowly
            lr *= 0.998
            # Increase beta to sharpen the soft-max over time
            beta *= 1.005
            if beta > 100.0:
                beta = 100.0
        else:
            stall_count += 1
            # Reduce learning rate
            lr *= 0.92
            if stall_count > max_stall:
                # Perturb the worst facets' vertices
                try:
                    hull = ConvexHull(current_points)
                    facets = hull.simplices
                    radii, centers, valid = _facet_circumradii(current_points, facets)
                    # Find worst facet
                    worst_idx = 0
                    worst_r = 0
                    for idx in range(len(facets)):
                        if not valid[idx]:
                            continue
                        if radii[idx] > worst_r:
                            worst_r = radii[idx]
                            worst_idx = idx
                    tri = facets[worst_idx]
                    for vi in tri:
                        noise = np.random.normal(0, 0.01, 3)
                        current_points[vi] += noise
                    current_points = _normalize(current_points)
                    stall_count = 0
                    lr = max(lr, 0.001)
                except Exception:
                    pass
            if lr < min_lr:
                break
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return best_points, best_radius

def _soft_min_pairwise(points, beta=20.0):
    """
    Compute a smooth surrogate for the minimum pairwise angular distance.
    This is used to maximize the minimum distance (Tammes problem variant).
    Returns (objective_value, gradient) where objective is the negative of the soft-min.
    We want to minimize -soft_min, which is equivalent to maximizing soft_min.
    """
    # Compute all pairwise angular distances
    D = points @ points.T
    D = np.clip(D, -1.0, 1.0)
    dists = np.arccos(D)
    
    # Set diagonal to inf so it doesn't affect the min
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min: -log-sum-exp(-dists / beta) / beta
    # To maximize soft-min, we minimize -soft-min = log-sum-exp(-dists / beta) / beta
    # Let's just compute the gradient of the soft-min directly and maximize it.
    # Or, simpler: use log-sum-exp of -dists.
    
    neg_dists = -dists
    neg_dists_flat = neg_dists.flatten()
    
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    neg_dists_masked = neg_dists_flat[mask]
    
    max_neg = np.max(neg_dists_masked)
    shifted = neg_dists_masked - max_neg
    log_sum_exp = max_neg + np.log(np.sum(np.exp(shifted / beta))) * beta
    
    # Gradient of log_sum_exp w.r.t. points
    # d/dp_i log(sum exp(-d_jk / beta)) = sum_jk (1/beta) exp(-d_jk / beta) / sum exp(-d_jk / beta) * d(-d_jk)/dp_i
    # d(-d_jk)/dp_i = -d(d_jk)/dp_i
    # d(d_jk)/dp_i for j != k:
    # d_jk = arccos(p_j . p_k)
    # d/dp_i arccos(p_j . p_k) = 0 if i != j and i != k
    # If i == j: - (p_k - (p_j . p_k) p_j) / sqrt(1 - (p_j . p_k)^2)
    # If i == k: - (p_j - (p_j . p_k) p_k) / sqrt(1 - (p_j . p_k)^2)
    
    weights = np.exp(shifted / beta) / np.sum(np.exp(shifted / beta))
    
    # Reshape weights back to matrix, excluding diagonal
    W = np.zeros((N, N))
    W[mask] = weights
    W /= beta # The 1/beta factor
    
    grad = np.zeros_like(points)
    
    # This loop is O(N^2). For N=130, this is 16900 iterations, which is fine.
    for i in range(N):
        for j in range(N):
            if i == j or W[i, j] == 0:
                continue
            p_i = points[i]
            p_j = points[j]
            dot = np.dot(p_i, p_j)
            sin_d = np.sqrt(max(0.0, 1.0 - dot * dot))
            if sin_d < 1e-12:
                continue
            
            # d(-d_ij)/dp_i = (p_j - dot * p_i) / sin_d
            # d(-d_ij)/dp_j = (p_i - dot * p_j) / sin_d
            
            w = W[i, j]
            
            grad_i = w * (p_j - dot * p_i) / sin_d
            grad_j = w * (p_i - dot * p_j) / sin_d
            
            # Project to tangent space
            grad_i -= np.dot(grad_i, p_i) * p_i
            grad_j -= np.dot(grad_j, p_j) * p_j
            
            grad[i] += grad_i
            grad[j] += grad_j
            
    # We want to maximize soft-min, so we minimize -soft-min.
    # The gradient we computed is for log-sum-exp(-dists), which is proportional to -soft-min.
    # So we return this gradient, and the caller will do points - lr * grad.
    # Wait, if we maximize soft-min, we should move in the direction of the gradient of soft-min.
    # soft-min = -log-sum-exp(-dists) / beta
    # grad(soft-min) = -grad(log-sum-exp(-dists)) / beta
    # So if we return -grad, the caller doing points - lr * (-grad) = points + lr * grad will move in the direction of increasing soft-min.
    # Let's just return the gradient of the objective we want to minimize.
    # Objective to minimize: -soft_min = log-sum-exp(-dists) / beta
    # Gradient of objective: grad(log-sum-exp(-dists)) / beta
    # This is exactly what we computed as `grad`.
    # So we return `grad`, and the caller does points - lr * grad.
    # This will move points in the direction of decreasing -soft_min, i.e., increasing soft_min.
    
    return float(log_sum_exp), grad

def _maximize_min_distance(points, start_time, deadline, initial_lr=0.01, beta=20.0):
    """
    Maximize the minimum pairwise distance using gradient ascent on a soft-min surrogate.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    # Use negative of min distance as proxy for "best"
    best_min_dist = -np.pi
    lr = initial_lr
    
    while time.time() < deadline:
        obj_val, grad = _soft_min_pairwise(current_points, beta)
        
        # Project gradient onto tangent space
        radial = np.einsum('ij,ij->i', grad, current_points)
        grad -= radial[:, None] * current_points
        
        # Gradient ascent: points + lr * grad
        new_points = current_points + lr * grad
        new_points = _normalize(new_points)
        
        # Evaluate actual min distance
        D = new_points @ new_points.T
        D = np.clip(D, -1.0, 1.0)
        dists = np.arccos(D)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = new_points.copy()
            current_points = new_points.copy()
            lr *= 0.999
        else:
            lr *= 0.95
            if lr < 1e-5:
                break
        
        if time.time() > deadline - 2:
            break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin

    # Phase 0: Generate diverse initializations
    rng = np.random.default_rng(42)
    
    # 1. Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    fib_pts = _normalize(fib_pts)
    
    # 2. Electrostatic refinement of Fibonacci
    def electrostatic_refine(pts, n_iters=100, lr=0.01):
        pts = pts.copy()
        for _ in range(n_iters):
            D = pts @ pts.T
            chord2 = 2.0 - 2.0 * D
            np.fill_diagonal(chord2, 1.0)
            W = 1.0 / chord2
            np.fill_diagonal(W, 0.0)
            row_sum = W.sum(axis=1)
            forces = pts * row_sum[:, None] - (W @ pts)
            forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            forces /= norms
            pts = pts + lr * forces
            pts = _normalize(pts)
        return pts
    
    if deadline - time.time() > 10.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=200, lr=0.005)
    else:
        elec_pts = fib_pts
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in [fib_pts, elec_pts]:
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: Maximize minimum pairwise distance (Tammes variant)
    # This creates a highly uniform distribution, which is a good starting point for covering.
    if deadline - time.time() > 20.0:
        tammes_deadline = time.time() + min(30.0, deadline - time.time() - 20.0)
        tammes_pts = _maximize_min_distance(best_points, start_time, tammes_deadline, initial_lr=0.005, beta=30.0)
        
        tammes_rad = _covering_radius_rad(tammes_pts)
        if tammes_rad < best_radius:
            best_radius = tammes_rad
            best_points = tammes_pts.copy()
    
    # Phase 2: Lloyd's algorithm for initial convergence
    points = best_points.copy()
    lloyd_deadline = time.time() + min(40.0, deadline - time.time() - 10.0)
    stall_count = 0
    max_stall = 15
    
    while time.time() < lloyd_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            points = new_points.copy()
            stall_count += 1
            
            if stall_count > max_stall:
                # Perturb a random subset
                num_perturb = max(5, N // 10)
                indices = np.random.choice(N, num_perturb, replace=False)
                noise = np.random.normal(0, 0.02, (num_perturb, 3))
                points[indices] += noise
                points[indices] = _normalize(points[indices])
                stall_count = 0
    
    # Phase 3: Gradient descent with smooth surrogate
    if deadline - time.time() > 20.0:
        grad_deadline = time.time() + min(60.0, deadline - time.time() - 5.0)
        points = best_points.copy()
        gd_points, gd_radius = _gradient_descent(points, start_time, grad_deadline, 
                                                initial_lr=0.003, min_lr=1e-7, initial_beta=30.0)
        
        if gd_radius < best_radius:
            best_radius = gd_radius
            best_points = gd_points.copy()
    
    # Phase 4: Final high-precision polish with Lloyd's + targeted perturbation
    if deadline - time.time() > 5.0:
        polish_deadline = deadline - 2.0
        points = best_points.copy()
        stall_count = 0
        max_stall = 20
        
        while time.time() < polish_deadline:
            new_points = _lloyd_iteration(points)
            new_points = _normalize(new_points)
            
            current_radius = _covering_radius_rad(new_points)
            
            if current_radius < best_radius - 1e-12:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points.copy()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > max_stall:
                    # Targeted perturbation of the worst facet's vertices
                    try:
                        hull = ConvexHull(points)
                        facets = hull.simplices
                        radii, centers, valid = _facet_circumradii(points, facets)
                        worst_idx = 0
                        worst_r = 0
                        for idx in range(len(facets)):
                            if not valid[idx]:
                                continue
                            if radii[idx] > worst_r:
                                worst_r = radii[idx]
                                worst_idx = idx
                        tri = facets[worst_idx]
                        for vi in tri:
                            noise = np.random.normal(0, 0.005, 3)
                            points[vi] += noise
                        points = _normalize(points)
                    except Exception:
                        pass
                    stall_count = 0
                    if time.time() > polish_deadline - 3:
                        break
    
    return best_points

