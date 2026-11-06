"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the radius score for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _fibonacci_sphere_init(n, radius_scale=1.0):
    """Generate points using Fibonacci sphere distribution, scaled to fit."""
    points = np.zeros((n, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n):
        z = 1.0 - (i + 0.5) / n
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x * radius_scale, y * radius_scale]
    
    return points


def _hexagonal_init(n):
    """Generate points using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98


def _force_relaxation(points, max_iters=2000, tol=1e-10, deadline=None):
    """
    Optimize point positions using force-directed relaxation.
    Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    """
    points = points.copy()
    n = len(points)
    
    # Current best radius
    best_r = _compute_score(points)
    current_r = best_r
    
    # Adaptive step size
    step_size = 0.05
    min_step = 1e-8
    max_step = 0.1
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: push apart close pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-12:
                    # Coincident points, add small random perturbation
                    forces[i] += np.random.randn(2) * 1e-6
                    forces[j] -= np.random.randn(2) * 1e-6
                    continue
                # Repulsive force: stronger when closer
                force_mag = 1.0 / (dist * dist + 1e-10)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Attraction to boundary: push points outward to maximize boundary distance
        for i in range(n):
            radius = np.sqrt(np.sum(points[i] ** 2))
            if radius > 1e-12:
                # Force pushing outward, stronger when closer to center
                force_mag = (1.0 - radius) * 0.5
                forces[i] += force_mag * points[i] / radius
            else:
                forces[i] += np.random.randn(2) * 0.01
        
        # Normalize forces
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1, keepdims=True))
        force_norms = np.maximum(force_norms, 1e-10)
        forces_normalized = forces / force_norms
        
        # Update positions
        new_points = points + step_size * forces_normalized
        
        # Keep points inside unit circle
        new_radii = np.sqrt(np.sum(new_points ** 2, axis=1))
        for i in range(n):
            if new_radii[i] > 1.0:
                new_points[i] = points[i] * (1.0 / new_radii[i])
        
        # Compute new score
        new_r = _compute_score(new_points)
        
        if new_r > current_r + 1e-12:
            # Accept, increase step size slightly
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
            step_size = min(step_size * 1.05, max_step)
        else:
            # Reject, decrease step size
            step_size = max(step_size * 0.95, min_step)
        
        # Check convergence
        if step_size < min_step * 1.01:
            break
    
    return points, best_r


def _gradient_ascent(points, max_iters=5000, tol=1e-12, deadline=None):
    """
    Gradient ascent on the smooth surrogate of the min objective.
    Uses soft-min approximation for smoothness.
    """
    points = points.copy()
    n = len(points)
    
    best_r = _compute_score(points)
    current_r = best_r
    
    # Soft-min temperature
    beta = 100.0  # Higher = closer to true min
    step_size = 0.01
    min_step = 1e-8
    max_step = 0.05
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute soft-min of pairwise distances and boundary distances
        # For pairwise: f(i,j) = dist_ij / 2
        # For boundary: g(i) = 1 - |c_i|
        
        # Approximate gradient of soft-min
        forces = np.zeros_like(points)
        
        # Pairwise contributions
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-12:
                    continue
                # Soft-min weight: exponential of negative scaled distance
                # We want to maximize the minimum, so we push apart the closest pairs
                weight = np.exp(-beta * dist / 2.0)
                # Gradient contribution: push i and j apart
                forces[i] += weight * diff / dist
                forces[j] -= weight * diff / dist
        
        # Boundary contributions
        for i in range(n):
            radius = np.sqrt(np.sum(points[i] ** 2))
            if radius > 1e-12:
                boundary_dist = 1.0 - radius
                weight = np.exp(-beta * boundary_dist)
                # Push outward
                forces[i] += weight * points[i] / radius
        
        # Normalize and update
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1, keepdims=True))
        force_norms = np.maximum(force_norms, 1e-10)
        forces_normalized = forces / force_norms
        
        new_points = points + step_size * forces_normalized
        
        # Keep inside unit circle
        new_radii = np.sqrt(np.sum(new_points ** 2, axis=1))
        for i in range(n):
            if new_radii[i] > 1.0:
                new_points[i] = points[i] * (1.0 / new_radii[i])
        
        new_r = _compute_score(new_points)
        
        if new_r > current_r + 1e-13:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
            step_size = min(step_size * 1.02, max_step)
        else:
            step_size = max(step_size * 0.97, min_step)
        
        if step_size < min_step * 1.01:
            break
    
    return points, best_r


def _polish(points, max_iters=10000, deadline=None):
    """
    High-precision polish pass with very tight tolerance.
    """
    points = points.copy()
    n = len(points)
    
    best_r = _compute_score(points)
    current_r = best_r
    
    step_size = 0.005
    min_step = 1e-10
    max_step = 0.02
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the critical constraint (minimum)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.sqrt(np.sum(points ** 2, axis=1))
        min_boundary = np.min(1.0 - radii)
        
        current_r = min(min_pair_dist / 2.0, min_boundary)
        
        if current_r > best_r:
            best_r = current_r
        
        # Compute forces focusing on critical constraints
        forces = np.zeros_like(points)
        
        # Pairwise: focus on pairs close to minimum
        for i in range(n):
            for j in range(i + 1, n):
                dist = dists[i, j]
                if dist < min_pair_dist * 1.5:
                    diff = points[i] - points[j]
                    if dist > 1e-12:
                        weight = 1.0 / (dist + 1e-10)
                        forces[i] += weight * diff / dist
                        forces[j] -= weight * diff / dist
        
        # Boundary: focus on points close to boundary
        for i in range(n):
            boundary_dist = 1.0 - radii[i]
            if boundary_dist < min_boundary * 1.5:
                if radii[i] > 1e-12:
                    weight = 1.0 / (boundary_dist + 1e-10)
                    forces[i] += weight * points[i] / radii[i]
        
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1, keepdims=True))
        force_norms = np.maximum(force_norms, 1e-10)
        forces_normalized = forces / force_norms
        
        new_points = points + step_size * forces_normalized
        
        # Keep inside unit circle
        new_radii = np.sqrt(np.sum(new_points ** 2, axis=1))
        for i in range(n):
            if new_radii[i] > 1.0:
                new_points[i] = points[i] * (1.0 / new_radii[i])
        
        new_r = _compute_score(new_points)
        
        if new_r > current_r + 1e-14:
            points = new_points
            step_size = min(step_size * 1.01, max_step)
        else:
            step_size = max(step_size * 0.98, min_step)
        
        if step_size < min_step * 1.01:
            break
    
    return points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal lattice
    hex_pts = _hexagonal_init(N)
    inits.append(hex_pts)
    
    # Fibonacci sphere at different scales
    for scale in [0.9, 0.95, 1.0]:
        fib_pts = _fibonacci_sphere_init(N, radius_scale=scale)
        inits.append(fib_pts)
    
    # Random uniform in disk
    for _ in range(3):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        random_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(random_pts)
    
    best_points = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Force relaxation
        relaxed_pts, relaxed_score = _force_relaxation(init, max_iters=1500, deadline=deadline - 20.0)
        
        # Gradient ascent
        grad_pts, grad_score = _gradient_ascent(relaxed_pts, max_iters=3000, deadline=deadline - 15.0)
        
        # Polish
        polished_pts, polished_score = _polish(grad_pts, max_iters=5000, deadline=deadline - 5.0)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
        
        if time.time() > deadline - 50.0:
            break
    
    # Final polish on best
    if best_points is not None and time.time() < deadline - 5.0:
        final_pts, final_score = _polish(best_points, max_iters=8000, deadline=deadline - 2.0)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score
    
    return best_points
# EVOLVE-BLOCK-END
