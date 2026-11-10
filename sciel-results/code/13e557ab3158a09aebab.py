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
    # Pairwise distances using vectorized computation
    dx = centers[:, 0:1] - centers[0:1, 0].T
    dy = centers[:, 1:2] - centers[0:2, 1].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r


def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
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


def _anneal(centers, start_temp=0.02, end_temp=0.00005, n_steps=30000, deadline=None):
    """Simulated annealing to improve the packing with adaptive step size."""
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    temp = start_temp
    step_size = 0.08
    
    for i in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a random perturbation
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.exponential(step_size)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_center = current.copy()
        new_center[idx, 0] += dx
        new_center[idx, 1] += dy
        
        # Keep inside container
        r = np.hypot(new_center[idx, 0], new_center[idx, 1])
        if r > 0.999:
            new_center[idx] *= 0.999 / r
        
        new_r = _compute_radius(new_center)
        
        if new_r > current_r:
            current = new_center
            current_r = new_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        else:
            # Accept worse solutions with probability exp((new_r - current_r) / temp)
            if new_r > 0 and np.random.random() < np.exp((new_r - current_r) / temp):
                current = new_center
                current_r = new_r
        
        # Cool down and reduce step size
        progress = i / n_steps
        temp = start_temp * (end_temp / start_temp) ** progress
        step_size = 0.08 * (0.01) ** progress
    
    return best, best_r


def _polish(centers, max_time=10.0):
    """Final polishing pass: move circles that are limiting the radius."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    while time.time() - start_time < max_time:
        # Find the circle that is limiting the radius
        dists_boundary = 1.0 - np.hypot(current[:, 0], current[:, 1])
        min_boundary_idx = np.argmin(dists_boundary)
        
        # Find pairs with minimum distance
        dx = current[:, 0:1] - current[0:1, 0].T
        dy = current[:, 1:2] - current[0:2, 1].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Try to move one of the limiting circles
        candidates = [min_boundary_idx, min_pair[0], min_pair[1]]
        for idx in candidates:
            if time.time() - start_time >= max_time:
                break
            
            # Try small random moves
            for _ in range(5):
                angle = np.random.uniform(0, 2 * np.pi)
                magnitude = np.random.uniform(0.001, 0.005)
                dx = magnitude * np.cos(angle)
                dy = magnitude * np.sin(angle)
                
                new_center = current.copy()
                new_center[idx, 0] += dx
                new_center[idx, 1] += dy
                
                r = np.hypot(new_center[idx, 0], new_center[idx, 1])
                if r > 0.999:
                    new_center[idx] *= 0.999 / r
                
                new_r = _compute_radius(new_center)
                
                if new_r > current_r:
                    current = new_center
                    current_r = new_r
                    if current_r > best_r:
                        best = current.copy()
                        best_r = current_r
                    break
                elif new_r > 0 and np.random.random() < 0.3:
                    current = new_center
                    current_r = new_r
    
    return best, best_r


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Main annealing phase
    best_centers, best_r = _anneal(centers, start_temp=0.02, end_temp=0.00005, 
                                    n_steps=40000, deadline=deadline)
    
    # If we have time, do additional annealing with different parameters
    while time.time() < deadline - 15.0:
        for temp in [0.01, 0.015, 0.008]:
            if time.time() >= deadline - 15.0:
                break
            new_centers, new_r = _anneal(best_centers, start_temp=temp, end_temp=0.00003,
                                         n_steps=15000, deadline=deadline - 15.0)
            if new_r > best_r:
                best_centers = new_centers
                best_r = new_r
    
    # Final polishing pass
    if time.time() < deadline - 5.0:
        best_centers, best_r = _polish(best_centers, max_time=min(10.0, deadline - time.time() - 2.0))
    
    return best_centers
# EVOLVE-BLOCK-END
