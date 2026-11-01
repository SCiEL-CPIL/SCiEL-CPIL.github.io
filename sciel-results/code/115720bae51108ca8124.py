"""Optimized covering of the unit sphere with 130 equal caps."""
import numpy as np
import time
import sys

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Uses the convex hull facets: the covering radius is the maximum circumradius
    of the triangular facets of the convex hull.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            c = n / norm_n
            cos_r = np.dot(c, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_circumradius:
                max_circumradius = r
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback: dense grid
        n_theta = 720
        n_phi = 360
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, np.pi, n_phi)
        Theta, Phi = np.meshgrid(theta, phi)
        x = np.sin(Phi) * np.cos(Theta)
        y = np.sin(Phi) * np.sin(Theta)
        z = np.cos(Phi)
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        max_dist = 0.0
        for i in range(0, len(grid_points), 50):
            batch = grid_points[i:i+50]
            dots = batch @ points.T
            cos_dists = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(cos_dists)
            min_dists = np.min(dists, axis=1)
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        return np.degrees(max_dist)


def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _compute_face_circumradii(points):
    """
    Compute circumradii for all facets of the convex hull.
    Returns (max_circumradius, worst_simplex, all_circumradii, all_simplices)
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    max_circumradius = 0.0
    worst_simplex = None
    all_circumradii = []
    all_simplices = []
    
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v1 = p1 - p2
        v2 = p1 - p3
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            all_circumradii.append(0.0)
            all_simplices.append(simplex)
            continue
        c = n / norm_n
        cos_r = np.dot(c, p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        all_circumradii.append(r)
        all_simplices.append(simplex)
        if r > max_circumradius:
            max_circumradius = r
            worst_simplex = simplex
    
    return max_circumradius, worst_simplex, np.array(all_circumradii), all_simplices


def _repulsion_step(points, step_size, time_budget):
    """
    Perform a repulsion-based optimization step.
    Points repel each other with a force that is stronger for closer points.
    Additionally, points in the worst facet are pushed toward the circumcenter.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Compute pairwise distances
    # Use dot products for efficiency
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Repulsion force: F_ij = 1 / (d_ij^3 + epsilon) for d_ij < threshold
    threshold = 0.5  # radians, ~28.6 degrees
    mask = dists < threshold
    np.fill_diagonal(mask, False)
    
    forces = np.zeros_like(points)
    for i in range(n):
        neighbors = np.where(mask[i])[0]
        if len(neighbors) == 0:
            continue
        d = dists[i, neighbors]
        # Force magnitude: stronger for closer points
        f = 1.0 / (d**3 + 1e-6)
        # Direction: from neighbor to point i
        for j_idx, j in enumerate(neighbors):
            # Direction is along the great circle from p_j to p_i
            # Approximate: p_i - (p_i·p_j)*p_j, normalized
            proj = np.dot(points[i], points[j]) * points[j]
            direction = points[i] - proj
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                forces[i] += f[j_idx] * direction
    
    # Apply forces
    points += step_size * forces
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def _targeted_optimization(points, max_time):
    """
    Targeted optimization: repeatedly find the worst facet and move its vertices
    to reduce the circumradius.
    """
    start_time = time.time()
    points = points.copy()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    step_size = 0.02
    decay = 0.999
    min_step = 1e-6
    
    while time.time() - start_time < max_time:
        try:
            max_cr, worst_simplex, all_cr, all_simplices = _compute_face_circumradii(points)
            
            if worst_simplex is None:
                break
            
            # Move the vertices of the worst facet toward the circumcenter
            p1, p2, p3 = points[worst_simplex[0]], points[worst_simplex[1]], points[worst_simplex[2]]
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                break
            c = n / norm_n
            
            # Move each vertex toward the circumcenter
            for idx in worst_simplex:
                # Direction toward c
                direction = c - np.dot(points[idx], c) * points[idx]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    points[idx] += step_size * direction
                    points[idx] = points[idx] / np.linalg.norm(points[idx])
            
            # Also apply a small global repulsion to maintain balance
            dots = points @ points.T
            np.fill_diagonal(dots, 1.0)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            mask = dists < 0.3
            np.fill_diagonal(mask, False)
            
            forces = np.zeros_like(points)
            for i in range(len(points)):
                neighbors = np.where(mask[i])[0]
                if len(neighbors) == 0:
                    continue
                d = dists[i, neighbors]
                f = 1.0 / (d**2 + 1e-4)
                for j_idx, j in enumerate(neighbors):
                    proj = np.dot(points[i], points[j]) * points[j]
                    direction = points[i] - proj
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                        forces[i] += f[j_idx] * direction
            
            points += (step_size * 0.1) * forces
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            # Check improvement
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            # Decay step size
            step_size *= decay
            if step_size < min_step:
                step_size = min_step
                
        except Exception:
            break
    
    return best_points, best_radius


def _multi_start_optimization(max_time):
    """
    Multi-start optimization with different initial configurations.
    """
    start_time = time.time()
    deadline = max_time - 5.0  # Safety margin
    
    best_points = None
    best_radius = float('inf')
    
    # Try multiple initial configurations
    n_starts = 5
    
    for s in range(n_starts):
        if time.time() - start_time > deadline:
            break
        
        # Generate different initial configurations
        if s == 0:
            points = _fibonacci_spiral(N)
        elif s == 1:
            # Random initial points
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif s == 2:
            # Fibonacci with slight perturbation
            points = _fibonacci_spiral(N)
            points += 0.1 * np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif s == 3:
            # Golden angle spiral with different phase
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.3)
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Another random start
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize this start
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 10.0:
            break
        
        try:
            optimized_points, radius = _targeted_optimization(points, remaining_time)
            if radius < best_radius:
                best_radius = radius
                best_points = optimized_points.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_spiral(N)
    
    return best_points


def construct():
    """
    Construct 130 points on the unit sphere that minimize the covering radius.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s limit
    
    # Use multi-start optimization
    try:
        points = _multi_start_optimization(deadline - start_time)
    except Exception:
        # Fallback to Fibonacci
        points = _fibonacci_spiral(N)
    
    # Final check: ensure points are on the unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Verify no coincident points
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    if np.max(np.abs(dots)) > 0.999999:
        # Points are too close, add small perturbation
        points += 1e-4 * np.random.randn(N, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
