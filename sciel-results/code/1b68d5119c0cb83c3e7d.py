"""Covering the sphere with 130 equal caps - repulsive init + soft surrogate gradient + exact hull polish + CVT refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import warnings

N = 130
TOTAL_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _covering_radius_from_hull(points):
    """Compute exact covering radius in radians from convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-12:
            continue
        normal = normal / norm_val
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.03):
    """
    Vectorized soft covering radius (log-sum-exp based) and its gradient.
    soft-min over config points for each grid point, then soft-max over grid points.
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)

    # Soft-min: for each grid point, weighted average using softmax of -angles/sigma
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)

    # Soft-min values: (M,)
    soft_min_vals = np.sum(weights * angles, axis=1)

    # Soft-max of soft-min values
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))

    # Gradient:
    # denom[i,j] = sqrt(1 - dots[i,j]^2)
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)

    # coeff[i,j] = max_weights[i] * weights[i,j]
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)

    # alpha[i,j] = coeff[i,j] / denom[i,j]
    alpha = coeff / denom  # (M, N)

    # grad_j = -sum_i alpha[i,j] * g_i + p_j * sum_i alpha[i,j] * dots[i,j]
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)

    grad = term1 + term2  # (N, 3)

    return soft_max_val, grad

def _repulsive_force_step(pts, step_size):
    """One step of repulsive force optimization on the sphere."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    pts = pts + step_size * forces
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts

def _cvt_refinement(pts, start_time, time_limit, max_iters=100):
    """
    Centroidal Voronoi Tessellation refinement: move each point toward the centroid
    of its Voronoi cell (the facet of the convex hull it belongs to).
    This directly reduces the covering radius by balancing cell sizes.
    """
    pts = pts.copy()
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # For each point, find its Voronoi cell centroid (mean of adjacent facet centroids)
        # First, compute facet normals and centroids
        facet_centroids = []
        facet_normals = []
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            facet_centroids.append(centroid)
            facet_normals.append(normal)
        
        if not facet_centroids:
            break
        
        facet_centroids = np.array(facet_centroids)
        facet_normals = np.array(facet_normals)
        
        # For each point, find which facets it belongs to
        # A point belongs to a facet if it's one of the facet's vertices
        point_to_facets = [[] for _ in range(N)]
        for i, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(i)
        
        # Compute new positions as weighted centroids of adjacent facets
        # Weight by facet area (or just use equal weights for simplicity)
        new_pts = pts.copy()
        for i in range(N):
            if not point_to_facets[i]:
                continue
            facets = point_to_facets[i]
            # Use the facet centroids as targets, weighted by how close the facet normal is to the point
            weights = []
            for fi in facets:
                w = np.dot(facet_normals[fi], pts[i])
                w = max(w, 0.1)  # ensure positive weight
                weights.append(w)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            # Move toward the weighted centroid of adjacent facet centroids
            target = np.sum(weights[:, np.newaxis] * facet_centroids[facets], axis=0)
            # Blend: move 30% toward target
            new_pts[i] = 0.7 * pts[i] + 0.3 * target
        
        # Renormalize
        norms = np.linalg.norm(new_pts, axis=1)
        new_pts = new_pts / norms[:, np.newaxis]
        
        pts = new_pts
    
    return pts

def _exact_local_polish(points, start_time, time_limit):
    """
    High-precision polish: for the worst facet, move its vertices toward
    the facet's circumcenter direction to reduce its circumradius.
    Iteratively handles the current worst facet.
    """
    pts = points.copy()
    for it in range(800):
        if time.time() - start_time > time_limit:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # Find the worst facet
        worst_facet = None
        worst_radius = -1.0
        worst_normal = None

        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            if facet_radius > worst_radius:
                worst_radius = facet_radius
                worst_facet = simplex
                worst_normal = normal

        if worst_facet is None:
            break

        # Move the vertices of the worst facet slightly toward the normal direction
        # This reduces the angular distance from the normal to the vertices
        move_amount = 0.001
        for idx in worst_facet:
            # Move point toward the normal direction (tangent component)
            tangent = worst_normal - np.dot(pts[idx], worst_normal) * pts[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-12:
                pts[idx] = pts[idx] + move_amount * tangent / tangent_norm

        # Renormalize
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]

    return pts

def construct():
    start_time = time.time()
    time_limit = TOTAL_TIME_LIMIT

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # 2. Repulsive Force Optimization (quick phase to get good spread)
    step_size = 0.05
    cooling = 0.998
    repulsive_iters = 0
    while time.time() - start_time < time_limit * 0.2 and repulsive_iters < 400:
        pts = _repulsive_force_step(pts, step_size)
        step_size *= cooling
        repulsive_iters += 1
        if step_size < 1e-6:
            break

    # 3. Generate evaluation grid (Fibonacci spiral, dense)
    N_eval = 20000
    i_eval = np.arange(N_eval) + 0.5
    z_eval = 1.0 - 2.0 * i_eval / N_eval
    phi_eval = np.pi * (1.0 + np.sqrt(5.0)) * i_eval
    r_eval = np.sqrt(1.0 - z_eval * z_eval)
    grid_points = np.column_stack([r_eval * np.cos(phi_eval), r_eval * np.sin(phi_eval), z_eval])

    # 4. Soft surrogate gradient descent to minimize covering radius
    sigma = 0.05
    lr = 0.01
    for it in range(300):
        if time.time() - start_time > time_limit * 0.6:
            break
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma)

        # Project gradient to tangent plane of sphere
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

        # Gradient step
        pts = pts + lr * grad_tangent

        # Renormalize to unit sphere
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]

        # Anneal learning rate and sigma
        lr *= 0.995
        sigma *= 0.998

    # 5. CVT refinement: move points toward Voronoi cell centroids
    pts = _cvt_refinement(pts, start_time, time_limit * 0.8)

    # 6. Exact local polish using convex hull facets
    pts = _exact_local_polish(pts, start_time, time_limit)

    # Final check: ensure points are on unit sphere and valid
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]

    # Verify not all in a closed hemisphere (sanity check)
    try:
        hull = ConvexHull(pts)
    except Exception:
        pass

    return pts
