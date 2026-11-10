"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

# Constants
_DEADLINE = 142.0  # Safety margin before 150s kill
_N_SAMPLES = 20000  # Number of sample points for covering radius estimation

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_fibonacci_samples(n_samples):
    """Generate well-distributed sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _covering_radius(pts, samples):
    """
    Estimate covering radius by computing max angular distance from samples to nearest point.
    Returns value in degrees.
    """
    # Batch computation to avoid memory issues for large sample sets
    batch_size = 5000
    min_max_dot = 1.0  # Initialize to 1.0 (corresponds to 0 degrees)
    
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch = samples[start:end]
        dots = batch @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        batch_min = np.min(max_dots)
        if batch_min < min_max_dot:
            min_max_dot = batch_min
    
    radius_rad = np.arccos(np.clip(min_max_dot, -1.0, 1.0))
    return np.degrees(radius_rad)

def _local_optimize(pts, samples, max_iters=3000, time_limit=None, start_time=None):
    """
    Local optimization using gradient-like steps based on farthest assigned samples.
    """
    if start_time is None:
        start_time = time.time()
        
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, samples)
    
    lr = 0.03  # Learning rate
    
    for it in range(max_iters):
        if time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Compute nearest configuration point for each sample
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        nearest_idx = dots.argmax(axis=1)  # (M,)
        
        new_pts = pts.copy()
        
        # For each configuration point, find the farthest assigned sample
        for j in range(N):
            mask = (nearest_idx == j)
            if not np.any(mask):
                continue
            
            assigned_samples = samples[mask]
            dots_j = dots[mask, j]
            
            # Find the sample with minimum dot product (farthest)
            far_idx = np.argmin(dots_j)
            far_sample = assigned_samples[far_idx]
            
            # Move point j towards far_sample
            diff = far_sample - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[j] = pts[j] + lr * diff / diff_norm
        
        new_pts = _normalize(new_pts)
        
        # Evaluate
        new_radius = _covering_radius(new_pts, samples)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            lr = max(0.005, lr * 0.999)
        else:
            lr = max(0.003, lr * 0.95)
    
    return best_pts, best_radius

def _repulsive_dynamics(pts, samples, max_iters=2000, time_limit=None, start_time=None):
    """
    Repulsive force-based dynamics: points repel from each other, 
    with strength inversely proportional to local density.
    """
    if start_time is None:
        start_time = time.time()
        
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, samples)
    
    dt = 0.01  # Time step
    repulsion_strength = 0.5
    
    for it in range(max_iters):
        if time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.sqrt(2 * (1 - dots))  # Angular distances
        
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(N):
            # Repulsion from other points
            other_dists = dists[i].copy()
            other_dists[i] = np.inf
            # Weight by inverse square of distance (capped)
            weights = 1.0 / np.maximum(other_dists, 0.05) ** 2
            weights[i] = 0.0
            
            # Force direction: away from other points
            for j in range(N):
                if i == j:
                    continue
                direction = pts[i] - pts[j]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    forces[i] += weights[j] * direction / dir_norm
        
        # Also add attraction towards farthest uncovered regions
        # Compute nearest point for each sample
        dots_s = samples @ pts.T
        dots_s = np.clip(dots_s, -1.0, 1.0)
        nearest_idx = dots_s.argmax(axis=1)
        
        # For each point, add pull towards its farthest assigned sample
        for j in range(N):
            mask = (nearest_idx == j)
            if np.any(mask):
                assigned_samples = samples[mask]
                dots_j = dots_s[mask, j]
                far_idx = np.argmin(dots_j)
                far_sample = assigned_samples[far_idx]
                # Weak attraction
                forces[j] += 0.1 * (far_sample - pts[j])
        
        # Update positions
        pts = pts + dt * forces
        pts = _normalize(pts)
        
        # Evaluate
        new_radius = _covering_radius(pts, samples)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = pts.copy()
    
    return best_pts, best_radius

def _generate_initial_points(method='fibonacci', rng=None):
    """Generate initial point configurations using different methods."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    elif method == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        pts = rng.normal(size=(N, 3))
        return _normalize(pts)
    elif method == 'perturbed_fibonacci':
        base = _generate_initial_points('fibonacci', rng)
        if rng is None:
            rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.15, size=base.shape)
        return _normalize(base + noise)
    else:
        raise ValueError(f"Unknown method: {method}")

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses multiple strategies: Fibonacci init, repulsive dynamics, local optimization.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Generate fixed samples for evaluation
    samples = _generate_fibonacci_samples(_N_SAMPLES)
    
    # Track best result
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fibonacci + local optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        pts_init = _generate_initial_points('fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 10
        opt_pts, opt_radius = _local_optimize(pts_init, samples, max_iters=5000, 
                                               time_limit=max(5, remaining), 
                                               start_time=start_time)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Strategy 2: Repulsive dynamics from Fibonacci
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        pts_init = _generate_initial_points('fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 10
        opt_pts, opt_radius = _repulsive_dynamics(pts_init, samples, max_iters=3000,
                                                   time_limit=max(5, remaining),
                                                   start_time=start_time)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Strategy 3: Random starts + local optimization
    n_random_starts = 5
    for i in range(n_random_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        pts_init = _generate_initial_points('random', rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        opt_pts, opt_radius = _local_optimize(pts_init, samples, max_iters=3000,
                                               time_limit=remaining,
                                               start_time=start_time)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Strategy 4: Perturbed Fibonacci + local optimization
    for i in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        pts_init = _generate_initial_points('perturbed_fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        opt_pts, opt_radius = _local_optimize(pts_init, samples, max_iters=3000,
                                               time_limit=remaining,
                                               start_time=start_time)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Final polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 8 and best_pts is not None:
        remaining = deadline - elapsed - 3
        if remaining > 3:
            final_pts, final_radius = _local_optimize(best_pts, samples, max_iters=2000,
                                                       time_limit=remaining,
                                                       start_time=start_time)
            if final_radius < best_radius:
                best_pts = final_pts.copy()
                best_radius = final_radius
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _generate_initial_points('fibonacci', rng)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
