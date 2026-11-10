"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _fibonacci_spiral(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _facet_circumcenter(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c
    (unit vectors). Returns the normalized circumcenter direction, or None
    if the triangle is degenerate.
    """
    cross_bc = np.cross(b, c)
    cross_ca = np.cross(c, a)
    cross_ab = np.cross(a, b)
    
    norm_bc = np.linalg.norm(cross_bc)
    norm_ca = np.linalg.norm(cross_ca)
    norm_ab = np.linalg.norm(cross_ab)
    
    total = norm_bc + norm_ca + norm_ab
    if total < 1e-14:
        return None
    
    circum_dir = (a * norm_bc + b * norm_ca + c * norm_ab) / total
    circum_norm = np.linalg.norm(circum_dir)
    if circum_norm < 1e-14:
        return None
    return circum_dir / circum_norm


def _cover_radius_exact(pts):
    """
    Compute the exact covering radius using the convex hull facets.
    Returns (radius_in_radians, worst_facet_indices, circumcenter).
    """
    if len(pts) < 3:
        return np.pi, None, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use dense sampling
        n_samples = 5000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = samples @ pts.T
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        worst_idx = np.argmax(angles)
        return float(angles[worst_idx]), None, None
    
    max_angle = 0.0
    worst_facet = None
    worst_cc = None
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        cc = _facet_circumcenter(a, b, c)
        if cc is None:
            continue
        
        dot_a = np.dot(cc, a)
        dot_b = np.dot(cc, b)
        dot_c = np.dot(cc, c)
        
        min_dot = min(dot_a, dot_b, dot_c)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angle = np.arccos(min_dot)
        
        if angle > max_angle:
            max_angle = angle
            worst_facet = simplex
            worst_cc = cc
    
    return max_angle, worst_facet, worst_cc


def _cover_radius_fast(pts, n_samples=3000):
    """Fast covering radius estimate via dense sampling."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    dots = samples @ pts.T
    min_dots = np.min(dots, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    return float(np.max(angles))


def _force_based_optimize(pts, max_iters=10000, initial_step=0.01, time_budget=30.0):
    """Force-based optimization with repulsion from close points."""
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts, n_samples=3000)
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        dots = pts @ pts.T
        np.fill_diagonal(dots, 1.0)
        
        target_dot = np.cos(np.sqrt(4 * np.pi / N))
        diff = target_dot - dots
        repel_mask = diff < 0
        repel_mag = np.where(repel_mask, (-diff) ** 2, 0.0)
        
        row_sums = np.sum(repel_mag, axis=1)
        forces = pts * row_sums[:, np.newaxis] - repel_mag @ pts
        
        far_idx = np.argmin(dots, axis=1)
        antipodes = -pts[far_idx]
        forces += 0.005 * antipodes
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-15, 1.0, force_norms)
        forces = forces / force_norms
        
        new_pts = pts + step * forces
        new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts_norms = np.where(new_pts_norms < 1e-15, 1.0, new_pts_norms)
        new_pts = new_pts / new_pts_norms
        
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
            if it % 50 == 0 and step < initial_step:
                step = min(step * 1.02, initial_step)
        else:
            no_improve_count += 1
            step *= 0.97
            if step < 1e-8:
                break
        
        if no_improve_count > 80:
            break
    
    return pts, best_radius


def _facet_gradient_optimize(pts, max_iters=5000, initial_step=0.003, time_budget=25.0):
    """
    STRUCTURAL CHANGE: Optimize using the exact convex hull facet information.
    For the worst facet, move its three vertices toward the circumcenter
    to reduce the circumradius. Also consider moving nearby points.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_exact(pts)[0]
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        radius, worst_facet, worst_cc = _cover_radius_exact(pts)
        
        if worst_facet is None or worst_cc is None:
            # Fallback to sampling-based gradient
            n_samples = 2000
            i = np.arange(n_samples) + 0.5
            z = 1.0 - 2.0 * i / n_samples
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            dots = samples @ pts.T
            min_dots, min_idx = np.min(dots, axis=1)
            min_dots = np.clip(min_dots, -1.0, 1.0)
            angles = np.arccos(min_dots)
            worst_idx = np.argmax(angles)
            worst_sample = samples[worst_idx]
            nearest_pt_idx = min_idx[worst_idx]
            
            new_pts = pts.copy()
            new_pts[nearest_pt_idx] = pts[nearest_pt_idx] + step * worst_sample
            norm = np.linalg.norm(new_pts[nearest_pt_idx])
            if norm > 1e-15:
                new_pts[nearest_pt_idx] /= norm
        else:
            # Move the three vertices of the worst facet toward the circumcenter
            new_pts = pts.copy()
            facet_pts = [pts[worst_facet[0]], pts[worst_facet[1]], pts[worst_facet[2]]]
            
            for k, idx in enumerate(worst_facet):
                # Move vertex toward circumcenter
                direction = worst_cc - pts[idx]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-15:
                    direction /= dir_norm
                    new_pts[idx] = pts[idx] + step * direction
                    norm = np.linalg.norm(new_pts[idx])
                    if norm > 1e-15:
                        new_pts[idx] /= norm
            
            # Also nudge nearby points (within 60 degrees) away from the circumcenter
            # to create more space
            for j in range(N):
                if j in worst_facet:
                    continue
                dot_cc = np.dot(pts[j], worst_cc)
                if dot_cc > np.cos(np.pi / 3):  # within 60 degrees
                    # Push away from circumcenter slightly
                    direction = pts[j] - worst_cc
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-15:
                        direction /= dir_norm
                        new_pts[j] = pts[j] + 0.3 * step * direction
                        norm = np.linalg.norm(new_pts[j])
                        if norm > 1e-15:
                            new_pts[j] /= norm
        
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
            if it % 30 == 0 and step < initial_step:
                step = min(step * 1.03, initial_step)
        else:
            no_improve_count += 1
            step *= 0.97
            if step < 1e-9:
                break
        
        if no_improve_count > 60:
            break
    
    return pts, best_radius


def _hill_climbing(pts, max_iters=3000, initial_step=0.005, time_budget=20.0):
    """Hill climbing with random perturbations."""
    pts = pts.copy()
    best_radius = _cover_radius_fast(pts, n_samples=3000)
    step = initial_step
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, n_perturb, replace=False)
        
        old_pts = pts[indices].copy()
        perturbations = rng.randn(n_perturb, 3) * step
        new_pts = pts.copy()
        for i, idx in enumerate(indices):
            new_pts[idx] = old_pts[i] + perturbations[i]
            norm = np.linalg.norm(new_pts[idx])
            if norm > 1e-15:
                new_pts[idx] /= norm
        
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            if it % 20 == 0 and step < initial_step:
                step = min(step * 1.05, initial_step)
        else:
            step *= 0.98
            if step < 1e-9:
                break
    
    return pts, best_radius


def _simulated_annealing(pts, max_iters=2000, t_start=0.05, t_end=0.001, time_budget=20.0):
    """
    Simulated annealing: accept worse solutions with probability exp(-delta/T).
    This helps escape local minima that hill climbing gets stuck in.
    """
    pts = pts.copy()
    current_radius = _cover_radius_fast(pts, n_samples=3000)
    best_radius = current_radius
    best_pts = pts.copy()
    start_time = time.time()
    rng = np.random.RandomState(456)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Annealing schedule
        progress = it / max_iters
        temperature = t_start * (t_end / t_start) ** progress
        
        # Random perturbation
        n_perturb = rng.randint(1, 5)
        indices = rng.choice(N, n_perturb, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            perturbation = rng.randn(3) * temperature
            new_pts[idx] = pts[idx] + perturbation
            norm = np.linalg.norm(new_pts[idx])
            if norm > 1e-15:
                new_pts[idx] /= norm
        
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        delta = new_radius - current_radius
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-10)):
            pts = new_pts
            current_radius = new_radius
            
            if new_radius < best_radius - 1e-12:
                best_radius = new_radius
                best_pts = new_pts.copy()
    
    return best_pts, best_radius


def _multi_start_optimize(n_restarts=6, time_budget=40.0, seed=42):
    """
    Multi-start with different initial configurations and optimization pipelines.
    Each restart uses a different perturbation of the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    best_pts = None
    best_radius = np.pi
    
    for restart in range(n_restarts):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 15:
            break
        
        # Different initial configurations
        if restart == 0:
            base = _fibonacci_spiral(N)
        elif restart == 1:
            # Rotated Fibonacci
            base = _fibonacci_spiral(N)
            angle = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
            base = base @ rot.T
        elif restart == 2:
            # Spherical random
            base = rng.randn(N, 3)
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        else:
            # Perturbed Fibonacci
            base = _fibonacci_spiral(N)
            noise_scale = 0.05 + 0.1 * restart / n_restarts
            noise = rng.randn(N, 3) * noise_scale
            base = base + noise
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        
        # Phase 1: Force-based
        budget1 = min(12.0, remaining * 0.3)
        optimized, r1 = _force_based_optimize(base, max_iters=5000, initial_step=0.01, time_budget=budget1)
        
        # Phase 2: Facet gradient (STRUCTURAL)
        remaining = time_budget - (time.time() - start_time)
        if remaining > 10 and r1 < best_radius:
            budget2 = min(15.0, remaining * 0.35)
            optimized, r2 = _facet_gradient_optimize(optimized, max_iters=3000, initial_step=0.003, time_budget=budget2)
            if r2 < best_radius:
                best_pts = optimized
                best_radius = r2
            elif r1 < best_radius:
                best_pts = optimized
                best_radius = r1
        
        # Phase 3: Simulated annealing (NEW)
        remaining = time_budget - (time.time() - start_time)
        if remaining > 8 and r1 < best_radius + 0.01:
            budget3 = min(10.0, remaining * 0.4)
            optimized, r3 = _simulated_annealing(optimized, max_iters=1500, t_start=0.03, t_end=0.001, time_budget=budget3)
            if r3 < best_radius:
                best_pts = optimized
                best_radius = r3
        
        # Phase 4: Hill climbing
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5 and r1 < best_radius + 0.01:
            budget4 = min(8.0, remaining * 0.5)
            optimized, r4 = _hill_climbing(optimized, max_iters=1500, initial_step=0.002, time_budget=budget4)
            if r4 < best_radius:
                best_pts = optimized
                best_radius = r4
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts, n_samples=5000)
    
    # Phase 1: Force-based optimization from Fibonacci start
    remaining = deadline - 10 - time.time()
    if remaining > 15:
        budget = min(40.0, remaining * 0.35)
        optimized, opt_radius = _force_based_optimize(pts, max_iters=12000, initial_step=0.01, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 2: Facet gradient optimization (STRUCTURAL CHANGE)
    remaining = deadline - 10 - time.time()
    if remaining > 12:
        budget = min(30.0, remaining * 0.4)
        optimized, opt_radius = _facet_gradient_optimize(best_pts, max_iters=5000, initial_step=0.003, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 3: Simulated annealing (NEW)
    remaining = deadline - 10 - time.time()
    if remaining > 10:
        budget = min(20.0, remaining * 0.4)
        optimized, opt_radius = _simulated_annealing(best_pts, max_iters=2000, t_start=0.03, t_end=0.0005, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 4: Multi-start with different initial configurations
    remaining = deadline - 8 - time.time()
    if remaining > 20:
        budget = min(50.0, remaining * 0.6)
        optimized, opt_radius = _multi_start_optimize(n_restarts=6, time_budget=budget, seed=42)
        if optimized is not None and opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 5: Final facet gradient refinement
    remaining = deadline - 5 - time.time()
    if remaining > 8:
        budget = min(15.0, remaining * 0.6)
        optimized, opt_radius = _facet_gradient_optimize(best_pts, max_iters=3000, initial_step=0.002, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 6: Final hill climbing
    remaining = deadline - 3 - time.time()
    if remaining > 5:
        budget = min(10.0, remaining * 0.8)
        optimized, opt_radius = _hill_climbing(best_pts, max_iters=2000, initial_step=0.001, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate with convex hull to ensure no degenerate configurations
    try:
        hull = ConvexHull(best_pts)
        if hull.volume < 1e-6:
            best_pts = _fibonacci_spiral(N)
            best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    except Exception:
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Final time guard
    if time.time() - start_time > deadline:
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
