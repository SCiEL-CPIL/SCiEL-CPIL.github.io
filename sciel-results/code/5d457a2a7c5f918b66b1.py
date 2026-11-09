"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi, None
    
    worst_cos = 1.0
    worst_facet = None
    worst_circ = None
    
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


def _soft_covering_radius_and_grad(points, beta=50.0):
    """
    Soft-max surrogate for covering radius using log-sum-exp over all facets.
    Returns (soft_value, gradient w.r.t. points).
    
    The soft covering radius is:
        soft_R = (1/beta) * log(sum_f exp(beta * R_f))
    where R_f is the circumradius of facet f.
    
    Gradient: d(soft_R)/d(x_i) = sum_f w_f * d(R_f)/d(x_i)
    where w_f = exp(beta * R_f) / sum_g exp(beta * R_g)
    """
    points = np.asarray(points)
    n = points.shape[0]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    m = len(facets)
    if m == 0:
        return np.pi, np.zeros_like(points)
    
    # Compute R_f and gradient contributions for each facet
    # R_f = arccos(cos_r_f) where cos_r_f = p_f . a_f
    # dR_f/dx_i = -1/sqrt(1-cos^2) * d(cos_r)/dx_i for i in facet
    
    cos_vals = np.zeros(m)
    p_vals = np.zeros((m, 3))
    
    for idx, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            cos_vals[idx] = 1.0
            p_vals[idx] = a
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_vals[idx] = np.clip(p @ a, -1.0, 1.0)
        p_vals[idx] = p
    
    # Soft max
    shifted = cos_vals / beta
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_val = -beta * log_sum_exp  # We want to minimize -cos, so maximize cos
    
    # Actually, covering radius R = arccos(cos). We want to minimize R.
    # soft_R = (1/beta) * log(sum exp(beta * R_f))
    # But R_f = arccos(cos_f), so exp(beta * R_f) = exp(beta * arccos(cos_f))
    # This is complex. Let's use a different approach.
    
    # Simpler: minimize max R_f. Use soft-min of cos (since minimizing R = maximizing cos).
    # soft_min_cos = -(1/beta) * log(sum exp(-beta * cos_f))
    # d(soft_min_cos)/d(x_i) = sum_f w_f * d(-cos_f)/d(x_i)
    # where w_f = exp(-beta * cos_f) / sum_g exp(-beta * cos_g)
    
    neg_cos = -cos_vals
    shifted_neg = neg_cos / beta
    max_shifted_neg = np.max(shifted_neg)
    exp_neg = np.exp(shifted_neg - max_shifted_neg)
    log_sum_exp_neg = max_shifted_neg + np.log(np.sum(exp_neg))
    soft_min_cos = beta * log_sum_exp_neg
    
    # Weights
    w = exp_neg / np.sum(exp_neg)
    
    # Gradient w.r.t. points
    grad = np.zeros_like(points)
    
    for idx in range(m):
        if w[idx] < 1e-12:
            continue
        facet = facets[idx]
        p = p_vals[idx]
        cos_val = cos_vals[idx]
        
        # d(cos)/d(x_i) for i in facet
        # cos = p . a, where p depends on a, b, c
        # This is complex. For a small step, we can approximate.
        # 
        # Alternative: use the fact that for a triangle on the sphere,
        # the circumcenter direction p is the normalized cross product of (a-b) and (b-c).
        # 
        # For gradient computation, let's use a simpler approach:
        # The derivative of cos_r w.r.t. a is approximately p (since cos = p.a)
        # But p also depends on a, b, c.
        # 
        # For a practical approximation, we use:
        # d(cos)/d(a) ≈ p - (p.a) * a  (tangent component)
        # This is the projection of p onto the tangent space at a.
        
        for i, idx_pt in enumerate(facet):
            pt = points[idx_pt]
            # Tangent component of p at pt
            tangent_p = p - np.dot(p, pt) * pt
            # d(cos)/d(pt) ≈ -tangent_p (since we want to increase cos, move pt toward p)
            # But we're computing gradient of -cos, so:
            grad[idx_pt] += w[idx] * (-tangent_p)
    
    # Normalize gradient to unit sphere constraint
    # The gradient should be in the tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return soft_min_cos, grad


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


def _smooth_optimize(points, deadline=None, beta_start=50.0, beta_end=200.0, max_iters=100):
    """
    Use L-BFGS-B to optimize the soft covering radius surrogate.
    """
    points = points.copy()
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        val, grad = _soft_covering_radius_and_grad(pts, beta=beta_start)
        return val, grad.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'gtol': 1e-6}
        )
        if result.success or result.fun < 1e-2:
            new_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_pts = new_pts / norms
            return new_pts
    except Exception:
        pass
    
    return points


def _repulsion_optimize(points, max_iters=1000, deadline=None):
    """
    Optimize using pairwise repulsion to maximize minimum distance.
    """
    points = points.copy()
    best_radius, _ = _all_worst_facets(points, top_k=1)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        cos_dist = points @ points.T
        np.fill_diagonal(cos_dist, 2.0)
        ang_dist = np.arccos(np.clip(cos_dist, -1.0, 1.0))
        
        min_dists = np.min(ang_dist, axis=1)
        
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = ang_dist[i, j]
                if d < 0.2:
                    w = (0.2 - d) / 0.2
                    direction = points[i] - points[j]
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                        tangent = direction - np.dot(direction, points[i]) * points[i]
                        forces[i] += w * tangent
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        step = 0.01
        new_points = points + step * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius, _ = _all_worst_facets(new_points, top_k=1)
        if new_radius < best_radius - 1e-14:
            points = new_points
            best_radius = new_radius
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
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
        
        remaining = deadline - time.time() - 20.0
        if remaining < 10.0:
            break
        
        max_iters = min(3000, int(remaining * 30))
        optimized = _geometric_repair(points, max_iters=max_iters, deadline=deadline)
        
        if time.time() < deadline - 10.0:
            optimized = _polish(optimized, max_iters=1000, deadline=deadline)
        
        radius, _ = _all_worst_facets(optimized, top_k=1)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Phase 2: Smooth optimization
    if best_points is not None and time.time() < deadline - 20.0:
        smoothed = _smooth_optimize(best_points, deadline=deadline, beta_start=50.0, max_iters=50)
        radius, _ = _all_worst_facets(smoothed, top_k=1)
        if radius < best_radius:
            best_radius = radius
            best_points = smoothed.copy()
        
        # Polish
        if time.time() < deadline - 10.0:
            best_points = _polish(best_points, max_iters=1500, deadline=deadline)
            radius, _ = _all_worst_facets(best_points, top_k=1)
            best_radius = radius
    
    # Phase 3: Repulsion-based optimization
    if best_points is not None and time.time() < deadline - 15.0:
        repulsed = _repulsion_optimize(best_points, max_iters=500, deadline=deadline)
        radius, _ = _all_worst_facets(repulsed, top_k=1)
        if radius < best_radius:
            best_radius = radius
            best_points = repulsed.copy()
        
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
