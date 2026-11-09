"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius_and_worst_facet(points):
    """
    Compute the covering radius in radians and identify the worst facet.
    Returns (radius, worst_facet_indices, circumcenter).
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, None, None
    
    hull = ConvexHull(points)
    facets = hull.simplices  # (M, 3) array of indices
    
    min_cos = 2.0
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
        
        cos_r = p @ a
        cos_r = np.clip(cos_r, -1.0, 1.0)
        if cos_r < min_cos:
            min_cos = cos_r
            worst_facet = facet.copy()
            worst_circ = p.copy()
    
    if min_cos >= 2.0:
        # Fallback
        min_cos = -1.0
    
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    return radius, worst_facet, worst_circ


def _local_optimize(points, max_iters=5000, deadline=None):
    """
    Combined local optimization using geometric repair with adaptive step size.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facet(points)[0]
    no_improve = 0
    step = 0.1
    best_points = points.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        radius, worst_facet, worst_circ = _covering_radius_and_worst_facet(points)
        
        if worst_facet is None:
            break
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 100:
            # Reduce step size
            step *= 0.5
            no_improve = 0
            if step < 1e-8:
                break
        
        i, j, k = worst_facet
        
        # Move vertices toward circumcenter with adaptive step
        for idx in [i, j, k]:
            pt = points[idx]
            direction = worst_circ - pt
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                new_pt = pt + step * direction
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return best_points


def _refine_with_restarts(points, num_restarts=5, deadline=None):
    """
    Refine a configuration using multiple perturbation restarts.
    """
    best_points = points.copy()
    best_radius = _covering_radius_and_worst_facet(points)[0]
    rng = np.random.default_rng(12345)
    
    for restart in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb the best solution
        perturbation_scale = 0.15 / (restart + 1)
        perturbation = perturbation_scale * rng.standard_normal((N, 3))
        new_points = best_points + perturbation
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Optimize
        optimized = _local_optimize(new_points, max_iters=2000, deadline=deadline)
        
        radius, _, _ = _covering_radius_and_worst_facet(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    return best_points


def construct():
    t_start = time.time()
    deadline = t_start + 125.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = fib_points / np.linalg.norm(fib_points, axis=1, keepdims=True)
    initializations.append(fib_points)
    
    # 2. Random points
    for _ in range(3):
        pts = rng.standard_normal((N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts)
    
    # 3. Golden ratio spiral variant
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = 2.399963229728653 * i  # Different golden angle
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    gr_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    gr_points = gr_points / np.linalg.norm(gr_points, axis=1, keepdims=True)
    initializations.append(gr_points)
    
    # Optimize each initialization
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # First pass: local optimization
        optimized = _local_optimize(init_points, max_iters=2000, deadline=deadline - 10.0)
        
        radius, _, _ = _covering_radius_and_worst_facet(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Second pass: refine the best solution with restarts
    if best_points is not None and time.time() < deadline - 15.0:
        refined = _refine_with_restarts(best_points, num_restarts=5, deadline=deadline - 5.0)
        
        radius, _, _ = _covering_radius_and_worst_facet(refined)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # Final fine-tuning if time permits
    if best_points is not None and time.time() < deadline - 5.0:
        final = _local_optimize(best_points, max_iters=1000, deadline=deadline - 2.0)
        
        radius, _, _ = _covering_radius_and_worst_facet(final)
        
        if radius < best_radius:
            best_radius = radius
            best_points = final.copy()
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
    
    if best_points is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points
