"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    x = centers[:, 0]
    y = centers[:, 1]
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _anneal(centers, max_time):
    """Simulated Annealing to improve packing."""
    start_time = time.time()
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    T = 0.05
    alpha = 0.99995
    step_size = 0.05
    
    # Precompute indices for efficiency if needed, but N is small
    while time.time() - start_time < max_time:
        # Batch of iterations
        for _ in range(1000):
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, 2)
            new_centers = current_centers.copy()
            new_centers[i] += delta
            
            new_r = _compute_radius(new_centers)
            
            if new_r > current_r:
                current_centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
            else:
                # Accept with probability exp((new_r - current_r) / T)
                # Only if T is large enough to matter
                if T > 1e-6:
                    prob = np.exp((new_r - current_r) / T)
                    if np.random.rand() < prob:
                        current_centers = new_centers
                        current_r = new_r
                        
        T *= alpha
        # Reduce step size slowly
        step_size *= 0.999
        
        # Check time
        if time.time() - start_time > max_time:
            break
            
    return best_centers, best_r

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 140.0 # 10s buffer before 150s kill
    
    # 1. Generate initial configuration
    current_centers = _hexagonal_packing()
    
    # 2. Anneal
    # Use most of the available time
    max_anneal_time = time_limit - (time.time() - start_time) - 5.0
    if max_anneal_time < 1.0:
        max_anneal_time = 1.0
        
    best_centers, best_r = _anneal(current_centers, max_anneal_time)
    
    # 3. Final check
    final_r = _compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to initial
        return current_centers
    return best_centers
# EVOLVE-BLOCK-END
