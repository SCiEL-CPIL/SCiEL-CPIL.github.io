"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _cover_radius_fast(pts):
    """
    Fast covering radius estimate: for each point on the sphere (sampled),
    find the minimum angular distance to any configuration point.
    We use a dense sampling approach combined with the convex hull method.
    
    Actually, the most reliable fast method: the covering radius equals the
    maximum over all Delaunay triangles of the circumradius. But computing
    the convex hull is expensive. 
    
    Alternative fast approach: sample many points on the sphere and compute
    the max of min-distance. This is an upper bound but with enough samples
    it's very close to the true value.
    
    For the optimization loop, we use a hybrid: compute the convex hull
    periodically, but use a fast gradient-based step in between.
    """
    # Fast upper bound via sampling
    # Use a Fibonacci spiral as sample points (dense enough for 50-point config)
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each sample to all config points
    # dots[i, j] = dot(samples[i], pts[j])
    dots = samples @ pts.T  # (n_samples, N)
    min_dots = np.min(dots, axis=1)  # (n_samples,)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    return np.max(angles)


def _cover_radius_hull(pts):
    """
    Compute the exact covering radius using the convex hull facets.
    For each triangular facet, compute the circumcenter and the angular distance
    to the nearest vertex. The covering radius is the maximum over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _cover_radius_fast(pts)
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter using weighted average (approximation)
        # Weight each vertex by the sine of the opposite angle
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        
        norm_bc = np.linalg.norm(cross_bc)
        norm_ca = np.linalg.norm(cross_ca)
        norm_ab = np.linalg.norm(cross_ab)
        
        total = norm_bc + norm_ca + norm_ab
        if total < 1e-15:
            continue
        
        circum_dir = (a * norm_bc + b * norm_ca + c * norm_ab) / total
        circum_norm = np.linalg.norm(circum_dir)
        if circum_norm < 1e-15:
            continue
        circum_dir = circum_dir / circum_norm
        
        # Compute the minimum dot product from circumcenter to the three vertices
        dot_a = np.dot(circum_dir, a)
        dot_b = np.dot(circum_dir, b)
        dot_c = np.dot(circum_dir, c)
        
        min_dot = min(dot_a, dot_b, dot_c)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angle = np.arccos(min_dot)
        
        if angle > max_angle:
            max_angle = angle
    
    return max_angle


def _force_based_optimize(pts, max_iters=5000, initial_step=0.01, time_budget=30.0):
    """
    Force-based optimization: for each point, compute the net force from
    nearby points. Points that are too close together repel; points that
    leave large gaps attract.
    
    Uses a fast evaluation (sampling-based) for the main loop, and
    convex hull for final validation.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts)
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        # Time check
        if time.time() - start_time > time_budget:
            break
        
        # Compute forces based on local density
        # For each pair, if distance is too small, repel; if there's a large empty region, attract
        forces = np.zeros_like(pts)
        
        # Compute all pairwise dot products
        dots = pts @ pts.T  # (N, N)
        np.fill_diagonal(dots, 1.0)
        
        # Target spacing: for 50 points, ideal angular spacing is about
        # sqrt(4*pi/50) ≈ 0.502 radians ≈ 28.7 degrees
        # cos(28.7°) ≈ 0.877
        target_dot = np.cos(np.sqrt(4 * np.pi / N))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dots[i, j]
                # If points are too close (d > target_dot), repel
                # If points are too far (d < target_dot), this doesn't directly help
                # but the global effect of moving toward empty regions helps
                
                # Repulsion force: push i away from j if they're too close
                if d > target_dot:
                    # Force direction: from j to i (away from j)
                    force_mag = (d - target_dot) ** 2
                    forces[i] += force_mag * (pts[i] - pts[j])
        
        # Also add a gentle force toward the "center of mass" of the opposite hemisphere
        # to prevent clustering
        for i in range(N):
            # Find the point farthest from pts[i]
            far_idx = np.argmin(dots[i, :])
            # Gentle attraction to the antipode of the farthest point
            antipode = -pts[far_idx]
            forces[i] += 0.01 * antipode
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-15, 1.0, force_norms)
        forces = forces / force_norms
        
        # Apply forces
        new_pts = pts + step * forces
        # Normalize back to unit sphere
        new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts_norms = np.where(new_pts_norms < 1e-15, 1.0, new_pts_norms)
        new_pts = new_pts / new_pts_norms
        
        # Evaluate
        new_radius = _cover_radius_fast(new_pts)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
            # Slightly increase step if we're making good progress
            if it % 50 == 0 and step < initial_step:
                step = min(step * 1.02, initial_step)
        else:
            no_improve_count += 1
            step *= 0.97
            if step < 1e-8:
                break
        
        if no_improve_count > 100:
            break
    
    return pts, best_radius


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _hill_climbing(pts, max_iters=2000, initial_step=0.005, time_budget=20.0):
    """
    Hill climbing: try small random perturbations and keep the best.
    """
    pts = pts.copy()
    best_radius = _cover_radius_fast(pts)
    step = initial_step
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random subset of points to perturb
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, n_perturb, replace=False)
        
        old_pts = pts[indices].copy()
        
        # Apply random perturbations
        perturbations = rng.randn(n_perturb, 3) * step
        new_pts = pts.copy()
        for i, idx in enumerate(indices):
            new_pts[idx] = old_pts[i] + perturbations[i]
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_radius = _cover_radius_fast(new_pts)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            # Occasionally increase step
            if it % 20 == 0 and step < initial_step:
                step = min(step * 1.05, initial_step)
        else:
            step *= 0.98
            if step < 1e-9:
                break
    
    return pts, best_radius


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts)
    
    # Phase 1: Force-based optimization from Fibonacci start
    remaining = deadline - 5 - time.time()
    if remaining > 10:
        budget = min(40.0, remaining * 0.4)
        optimized, opt_radius = _force_based_optimize(pts, max_iters=8000, initial_step=0.01, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 2: Hill climbing refinement
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        budget = min(30.0, remaining * 0.5)
        optimized, opt_radius = _hill_climbing(best_pts, max_iters=5000, initial_step=0.003, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 3: Multiple random restarts with perturbed Fibonacci
    rng = np.random.RandomState(42)
    for restart in range(10):
        remaining = deadline - 5 - time.time()
        if remaining < 10:
            break
        
        # Generate perturbed Fibonacci
        base = _fibonacci_spiral(N)
        noise_scale = 0.05 + 0.1 * restart / 10
        noise = rng.randn(N, 3) * noise_scale
        random_pts = base + noise
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        # Quick force-based optimization
        budget = min(15.0, remaining * 0.3)
        optimized, opt_radius = _force_based_optimize(random_pts, max_iters=3000, initial_step=0.008, time_budget=budget)
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final refinement with hill climbing
    remaining = deadline - 3 - time.time()
    if remaining > 5:
        budget = min(20.0, remaining * 0.8)
        optimized, opt_radius = _hill_climbing(best_pts, max_iters=3000, initial_step=0.002, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate with convex hull to ensure no degenerate configurations
    try:
        hull = ConvexHull(best_pts)
        # Check that the configuration is not in a closed hemisphere
        # by verifying the convex hull has positive volume
        if hull.volume < 1e-6:
            best_pts = _fibonacci_spiral(N)
            best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    except Exception:
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
