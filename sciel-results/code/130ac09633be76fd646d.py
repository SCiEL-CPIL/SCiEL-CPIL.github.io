"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in radians using the convex hull.
    Returns the radius and the worst facet info.
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, None
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None
    
    facets = hull.simplices  # (M, 3) array of indices
    if len(facets) == 0:
        return np.pi, None
    
    # Compute circumcenter and cos_r for each facet
    worst_cos = 1.0
    worst_facet = None
    worst_circ = None
    
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        # Compute circumcenter direction
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_r = np.clip(p @ a, -1.0, 1.0)
        if cos_r < worst_cos:
            worst_cos = cos_r
            worst_facet = facet.copy()
            worst_circ = p.copy()
    
    if worst_facet is None:
        return np.pi, None
    
    radius = np.arccos(np.clip(worst_cos, -1.0, 1.0))
    return radius, (worst_facet, worst_circ)


def _all_worst_facets(points, top_k=5):
    """
    Compute covering radius and return top_k worst facets.
    Returns (radius, list of (facet_indices, circumcenter)).
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, []
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, []
    
    facets = hull.simplices
    m = len(facets)
    if m == 0:
        return np.pi, []
    
    facet_data = []
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_r = np.clip(p @ a, -1.0, 1.0)
        facet_data.append((cos_r, facet.copy(), p.copy()))
    
    if not facet_data:
        return np.pi, []
    
    facet_data.sort(key=lambda x: x[0])
    min_cos = facet_data[0][0]
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    top_facets = [(f[1], f[2]) for f in facet_data[:top_k]]
    return radius, top_facets


def _soft_max_surrogate(points, beta=20.0):
    """
    Fast O(N^2) soft-max surrogate for covering radius.
    For each point y on the sphere, the distance to nearest point is max over
    antipodal caps. We approximate the covering radius by considering the
    maximum over all pairs of the angular distance, softened with log-sum-exp.
    
    Actually, a better surrogate: for each pair of points, compute the angular
    distance. The covering radius is at least half the max gap in the spherical
    Voronoi diagram. A practical surrogate is the max over all points of the
    maximum angular distance to its k nearest neighbors, softened.
    
    Simplest effective surrogate: max over all pairs of angular distance,
    softened. This correlates with covering radius.
    """
    points = np.asarray(points)
    n = points.shape[0]
    
    # Compute all pairwise angular distances
    # cos_dist[i,j] = dot(points[i], points[j])
    cos_dist = points @ points.T
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    
    # Angular distance
    ang_dist = np.arccos(cos_dist)
    
    # Set diagonal to 0
    np.fill_diagonal(ang_dist, 0.0)
    
    # For each point, find its farthest neighbor - but that's not right.
    # Better: the covering radius is related to the largest empty cap.
    # A good surrogate: for each point, the max distance to its nearest neighbors
    # in a local sense. Let's use: for each point, the angular distance to its
    # 5th nearest neighbor, then take the max (softened).
    
    k = min(5, n - 1)
    # For each row, get the k smallest distances (excluding self)
    # ang_dist[i, i] = 0, so we need to exclude that
    sorted_dists = np.sort(ang_dist, axis=1)
    # k-th nearest is at index k (0-indexed, since index 0 is self)
    kth_dists = sorted_dists[:, k]
    
    # Soft max of kth_dists
    shifted = kth_dists / beta
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_val = beta * log_sum_exp
    
    # Gradient: d(soft_val)/d(kth_dists[i]) = exp(kth_dists[i]/beta) / sum(exp(kth_dists[j]/beta))
    weights = np.exp(kth_dists / beta)
    weights /= np.sum(weights)
    
    # Now compute gradient w.r.t. points
    # This is complex. Let's use a simpler approach: 
    # For the point with max kth distance, move it away from its neighbors.
    
    return soft_val, weights


def _geometric_repair(points, max_iters=3000, deadline=None, top_k=3):
    """
    Iteratively move vertices of worst facets toward their circumcenters.
    """
    points = points.copy()
    best_radius, _ = _all_worst_facets(points, top_k=1)
    no_improve = 0
    step = 0.06
    min_step = 1e-10
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        radius, top_facets = _all_worst_facets(points, top_k=top_k)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-13:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive step
        if no_improve > 30:
            step *= 0.75
            no_improve = 0
        elif no_improve > 10:
            step *= 0.9
        elif no_improve < 5 and step < 0.08:
            step = min(step * 1.05, 0.1)
        
        if step < min_step:
            break
        
        # Move vertices of worst facets toward circumcenters
        moved = set()
        for facet, circ in top_facets:
            for idx in facet:
                if idx in moved:
                    continue
                moved.add(idx)
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    new_pt = pt + step * direction
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    points[idx] = new_pt
    
    return points


def _polish(points, max_iters=2000, deadline=None):
    """
    Fine-tuning pass with very small steps.
    """
    points = points.copy()
    best_radius, _ = _all_worst_facets(points, top_k=1)
    step = 0.003
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        radius, top_facets = _all_worst_facets(points, top_k=2)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-14:
            best_radius = radius
            step *= 0.98
        else:
            step *= 0.5
        
        if step < 1e-12:
            break
        
        moved = set()
        for facet, circ in top_facets:
            for idx in facet:
                if idx in moved:
                    continue
                moved.add(idx)
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    new_pt = pt + step * direction
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    points[idx] = new_pt
    
    return points


def _repulsion_optimize(points, max_iters=1000, deadline=None):
    """
    Optimize using pairwise repulsion to maximize minimum distance,
    which correlates with covering radius improvement.
    """
    points = points.copy()
    best_radius, _ = _all_worst_facets(points, top_k=1)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute pairwise distances
        cos_dist = points @ points.T
        np.fill_diagonal(cos_dist, 2.0)
        ang_dist = np.arccos(np.clip(cos_dist, -1.0, 1.0))
        
        # For each point, find its nearest neighbor distance
        min_dists = np.min(ang_dist, axis=1)
        
        # Points with small min_dist should be pushed away
        # Force on point i: sum over j of direction away from j, weighted by closeness
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = ang_dist[i, j]
                if d < 0.2:  # Only consider close pairs
                    w = (0.2 - d) / 0.2
                    # Direction from j to i
                    direction = points[i] - points[j]
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                        # Project to tangent space
                        tangent = direction - np.dot(direction, points[i]) * points[i]
                        forces[i] += w * tangent
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Apply small step
        step = 0.01
        new_points = points + step * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if it improves
        new_radius, _ = _all_worst_facets(new_points, top_k=1)
        if new_radius < best_radius - 1e-14:
            points = new_points
            best_radius = new_radius
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Generate initial configurations
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_points():
        pts = rng.standard_normal((N, 3))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Phase 1: Multiple restarts with geometric repair
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral()
        elif restart == 1:
            points = random_points()
        else:
            if best_points is not None and rng.random() < 0.5:
                perturbation = 0.15 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = random_points()
        
        # Geometric repair
        remaining = deadline - time.time() - 20.0
        if remaining < 10.0:
            break
        
        max_iters = min(3000, int(remaining * 30))
        optimized = _geometric_repair(points, max_iters=max_iters, deadline=deadline)
        
        # Polish
        if time.time() < deadline - 10.0:
            optimized = _polish(optimized, max_iters=1000, deadline=deadline)
        
        radius, _ = _all_worst_facets(optimized, top_k=1)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Phase 2: Repulsion-based optimization on best
    if best_points is not None and time.time() < deadline - 15.0:
        repulsed = _repulsion_optimize(best_points, max_iters=500, deadline=deadline)
        radius, _ = _all_worst_facets(repulsed, top_k=1)
        if radius < best_radius:
            best_radius = radius
            best_points = repulsed.copy()
        
        # Polish again
        if time.time() < deadline - 8.0:
            best_points = _polish(best_points, max_iters=1500, deadline=deadline)
            radius, _ = _all_worst_facets(best_points, top_k=1)
            best_radius = radius
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _polish(best_points, max_iters=800, deadline=deadline)
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
    else:
        best_points = fibonacci_spiral()
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points
