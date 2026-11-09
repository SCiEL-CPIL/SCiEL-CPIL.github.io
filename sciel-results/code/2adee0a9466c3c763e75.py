# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd


def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def hexagonal_init(n):
    """Initialize points on a hexagonal grid pattern"""
    points = []
    rows = 4
    cols = 4
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    
    return np.array(points[:n])


def repulsive_relaxation(points, start_time, max_time=580.0):
    """
    Relax points using repulsive forces with cooling schedule.
    Points repel each other with inverse-square force, encouraging uniform spacing.
    """
    points = points.copy().astype(float)
    n = len(points)
    
    # Normalize initial scale
    dists = spd.pdist(points)
    current_dmax = np.max(dists)
    if current_dmax > 0:
        points = points / current_dmax
    
    # Parameters
    temp = 1.0
    cooling = 0.995
    min_temp = 1e-8
    max_iter = 100000
    
    for iteration in range(max_iter):
        # Check time budget
        if time.time() - start_time > max_time:
            break
        
        # Compute pairwise forces
        # Force on point i from point j: F = (r_j - r_i) / |r_j - r_i|^3 (repulsive)
        # We use a softer version to avoid singularities
        forces = np.zeros_like(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Jitter if coincident
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                # Repulsive force: proportional to 1/dist^2, direction away from other point
                force_mag = 1.0 / (dist * dist + 0.01)
                force_vec = diff / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Center of mass correction (keep centroid at origin)
        com = np.mean(points, axis=0)
        forces -= np.mean(forces, axis=0)
        
        # Update positions with temperature-controlled step
        step_size = temp
        points += forces * step_size
        
        # Keep points bounded (prevent explosion)
        # Normalize to keep dmax ~ 1
        dists = spd.pdist(points)
        dmax = np.max(dists)
        if dmax > 0:
            points = points / dmax
        
        # Cool down
        temp *= cooling
        if temp < min_temp:
            break
    
    return points


def local_refinement(points, start_time, max_time=580.0):
    """
    Local refinement using small random perturbations accepted if they improve score.
    """
    points = points.copy().astype(float)
    n = len(points)
    current_score = calculate_score(points)
    
    # Normalize
    dists = spd.pdist(points)
    dmax = np.max(dists)
    if dmax > 0:
        points = points / dmax
        current_score = calculate_score(points)
    
    temp = 0.01
    cooling = 0.99
    min_temp = 1e-10
    
    while time.time() - start_time < max_time:
        if temp < min_temp:
            break
        
        # Generate a random perturbation
        idx = np.random.randint(n)
        delta = np.random.randn(2) * temp
        new_points = points.copy()
        new_points[idx] += delta
        
        # Normalize
        dists = spd.pdist(new_points)
        dmax = np.max(dists)
        if dmax > 0:
            new_points = new_points / dmax
        
        new_score = calculate_score(new_points)
        
        # Accept if better, or with probability exp((new-old)/T) for worse
        if new_score >= current_score:
            points = new_points
            current_score = new_score
        else:
            prob = np.exp((new_score - current_score) / max(temp, 1e-10))
            if np.random.random() < prob:
                points = new_points
                current_score = new_score
        
        temp *= cooling
    
    return points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit
    n = 16
    
    # Phase 1: Hexagonal initialization + repulsive relaxation
    points = hexagonal_init(n)
    points = repulsive_relaxation(points, start_time, max_time - 120)
    
    best_points = points.copy()
    best_score = calculate_score(best_points)
    
    # Phase 2: Multiple restarts with different initializations + local refinement
    np.random.seed(42)
    
    num_restarts = 5
    restart_budget = (max_time - 120) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time - 60:
            break
        
        # Different initializations
        if r == 0:
            # Circle initialization
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif r == 1:
            # Random initialization
            points = np.random.randn(n, 2)
        elif r == 2:
            # Square grid
            points = hexagonal_init(n) + np.random.randn(n, 2) * 0.1
        elif r == 3:
            # Two concentric circles
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            c1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
            c2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
            points = np.vstack([c1, c2])
        else:
            # Random
            points = np.random.randn(n, 2)
        
        # Relax
        points = repulsive_relaxation(points, start_time, max_time - 120)
        
        # Local refinement
        points = local_refinement(points, start_time, max_time - 30)
        
        score = calculate_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points


# EVOLVE-BLOCK-END
