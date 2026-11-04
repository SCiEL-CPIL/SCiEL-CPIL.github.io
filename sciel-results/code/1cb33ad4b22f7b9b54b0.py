"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum over all facets of the angular distance
    from the facet's circumcenter (on the sphere) to the farthest vertex.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
        # For each triangular facet, compute the circumcenter on the sphere
        # and the angular distance to each vertex
        max_circumradius = 0.0
        
        for simplex in hull.simplices:
            # Get the three points forming this triangle
            tri = pts[simplex]  # shape (3, 3)
            
            # Compute the circumcenter of the spherical triangle
            v1 = tri[0] - tri[1]
            v2 = tri[1] - tri[2]
            c = np.cross(v1, v2)
            
            if np.linalg.norm(c) < 1e-10:
                # Degenerate case, use centroid
                c = np.mean(tri, axis=0)
            
            c = c / np.linalg.norm(c)
            
            # Compute angular distances from c to each vertex
            dots = np.dot(tri, c)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            circumradius = np.max(dists)
            
            if circumradius > max_circumradius:
                max_circumradius = circumradius
        
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback: dense sampling
        n_test = 20000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        max_dist = 0.0
        batch_size = 5000
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch = test_pts[start:end]
            dots = batch @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        return np.degrees(max_dist)


def _facet_circumradii(pts):
    """Compute circumradii for all facets of the convex hull.
    Returns (circumradii, facet_indices) where facet_indices[i] is the 3-tuple of point indices.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
        circumradii = []
        facet_indices = []
        
        for simplex in hull.simplices:
            tri = pts[simplex]
            
            v1 = tri[0] - tri[1]
            v2 = tri[1] - tri[2]
            c = np.cross(v1, v2)
            
            if np.linalg.norm(c) < 1e-10:
                c = np.mean(tri, axis=0)
            
            c = c / np.linalg.norm(c)
            
            dots = np.dot(tri, c)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            circumradius = np.max(dists)
            
            circumradii.append(circumradius)
            facet_indices.append(simplex)
        
        return np.array(circumradii), facet_indices
    except Exception:
        return np.array([]), []


def _smooth_objective_with_grad(flat_pts, n_test=5000):
    """Smooth objective with analytical gradient using test points.
    Objective: variance of nearest-neighbor distances + soft max penalty.
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    # Generate test points (Fibonacci spiral)
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Compute angular distances
    dots = test_pts @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_test, N)
    
    # For each test point, find nearest point and its distance
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    # Variance of distances (encourages equiangularity)
    mean_dist = np.mean(min_dists)
    var = np.mean((min_dists - mean_dist) ** 2)
    
    # Soft max to penalize large covering radius
    beta = 40.0
    exp_dists = np.exp(beta * min_dists)
    log_sum_exp = np.log(np.sum(exp_dists)) / beta
    
    # Combined objective
    obj = var + 0.15 * log_sum_exp
    
    # Compute gradient
    grad_pts = np.zeros_like(pts)
    
    # Gradient of variance term
    # d(var)/d(min_dists_i) = 2*(min_dists_i - mean_dist)/n_test
    d_var = 2.0 * (min_dists - mean_dist) / n_test
    
    # Gradient of soft max term
    # d(log_sum_exp)/d(min_dists_i) = exp(beta*min_dists_i) / sum(exp(beta*min_dists))
    d_lse = exp_dists / np.sum(exp_dists)
    
    # Combined gradient w.r.t. min_dists
    d_obj_dmin = d_var + 0.15 * d_lse
    
    # d(min_dists_i)/d(pts_j) = -d(dists_{i,j})/d(pts_j) if j is nearest to i
    # d(dists)/d(pts) = -dot / sqrt(1-dot^2) * (test_point - pts * dot)
    # More precisely: dists = arccos(test @ pts.T)
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(x)/d(pts_j) = -test_point (since x = test @ pts_j)
    # So d(dists)/d(pts_j) = test_point / sqrt(1 - (test@pts_j)^2)
    
    # Only contribute from nearest neighbors
    for i in range(n_test):
        j = nearest[i]
        t = test_pts[i]
        dot_val = dots[i, j]
        sqrt_term = np.sqrt(np.maximum(1.0 - dot_val * dot_val, 1e-12))
        # d(dists_i)/d(pts_j) = t / sqrt_term
        grad_pts[j] -= d_obj_dmin[i] * t / sqrt_term
    
    # Convert to flat gradient, accounting for normalization
    # The objective is computed on normalized pts, so we need to chain rule
    # For simplicity, use the gradient as-is (normalization is smooth near unit sphere)
    grad_flat = grad_pts.flatten()
    
    return obj, grad_flat


def _facet_refine(pts, n_iter=50, step=0.05):
    """Refine points by equalizing facet circumradii.
    Move points to reduce the variance of facet circumradii.
    """
    pts = _normalize(pts.copy())
    
    for iteration in range(n_iter):
        circumradii, facet_indices = _facet_circumradii(pts)
        
        if len(circumradii) == 0:
            break
        
        mean_cr = np.mean(circumradii)
        std_cr = np.std(circumradii)
        
        if std_cr < 1e-8:
            break
        
        # For each facet, compute gradient direction to reduce its circumradius
        # if it's above mean, or increase if below mean
        grad_pts = np.zeros_like(pts)
        
        for i, simplex in enumerate(facet_indices):
            tri = pts[simplex]
            cr = circumradii[i]
            
            # Weight: how much this facet deviates from mean
            weight = cr - mean_cr
            
            # Compute circumcenter
            v1 = tri[0] - tri[1]
            v2 = tri[1] - tri[2]
            c = np.cross(v1, v2)
            if np.linalg.norm(c) < 1e-10:
                c = np.mean(tri, axis=0)
            c = c / np.linalg.norm(c)
            
            # Move vertices toward/away from circumcenter based on weight
            for k, idx in enumerate(simplex):
                # Move point toward circumcenter if cr > mean (reduce), away if cr < mean
                direction = c - pts[idx]
                # Ensure direction is tangent to sphere
                direction = direction - np.dot(direction, pts[idx]) * pts[idx]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                grad_pts[idx] += weight * direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_pts)
        if grad_norm > 1e-10:
            grad_pts = grad_pts / grad_norm
        
        # Update points
        pts = pts + step * grad_pts
        pts = _normalize(pts)
        
        # Check convergence
        if iteration > 0:
            change = np.linalg.norm(pts - _normalize(pts - step * grad_pts / grad_norm))
            if change < 1e-10:
                break
    
    return pts


def _voronoi_centroid_refine(pts, n_test=15000, n_iter=50, step=0.05):
    """Refine points by moving them toward their Voronoi cell centroids."""
    pts = _normalize(pts.copy())
    
    # Generate test points
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    for iteration in range(n_iter):
        # Compute nearest neighbor for each test point
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        nearest = np.argmin(dists, axis=1)
        
        # Compute Voronoi cell centroids using bincount for efficiency
        new_pts = np.zeros_like(pts)
        for dim in range(3):
            sums = np.bincount(nearest, weights=test_pts[:, dim], minlength=N)
            counts = np.bincount(nearest, minlength=N)
            counts[counts == 0] = 1
            new_pts[:, dim] = sums / counts
        
        # Normalize centroids
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        new_pts = new_pts / norms
        
        # Blend with current points
        pts = (1 - step) * pts + step * new_pts
        pts = _normalize(pts)
        
        # Check convergence
        if iteration > 0:
            change = np.linalg.norm(pts - _normalize(new_pts))
            if change < 1e-10:
                break
    
    return pts


def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    best_pts = None
    best_radius = np.inf
    
    # Try multiple initializations
    n_restarts = 4
    
    for restart in range(n_restarts):
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random initialization with some structure
            rng = np.random.default_rng(42 + restart)
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.normal(0, 0.1, N)
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts0 = _normalize(pts0)
        elif restart == 2:
            # More aggressive random perturbation
            rng = np.random.default_rng(42 + restart)
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.normal(0, 0.3, N)
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts0 = _normalize(pts0)
        else:
            # Different Fibonacci variant
            rng = np.random.default_rng(100 + restart)
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.normal(0, 0.5, N)
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts0 = _normalize(pts0)
        
        # Step 1: Voronoi centroid refinement
        pts_refined = _voronoi_centroid_refine(pts0, n_test=10000, n_iter=30, step=0.1)
        
        # Step 2: Facet-based refinement to equalize circumradii
        pts_facet = _facet_refine(pts_refined, n_iter=30, step=0.05)
        
        # Step 3: Gradient-based optimization with analytical gradient
        flat0 = pts_facet.flatten()
        
        try:
            res = minimize(_smooth_objective_with_grad, flat0, method='L-BFGS-B',
                         jac=True, options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            pts_opt = _normalize(res.x.reshape(N, 3))
        except Exception:
            pts_opt = pts_facet
        
        # Step 4: Final facet refinement
        pts_final_facet = _facet_refine(pts_opt, n_iter=50, step=0.03)
        
        # Step 5: Final Voronoi refinement
        pts_final = _voronoi_centroid_refine(pts_final_facet, n_test=20000, n_iter=50, step=0.03)
        
        # Compute covering radius
        radius = _covering_radius(pts_final)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts_final
    
    return best_pts
# EVOLVE-BLOCK-END
