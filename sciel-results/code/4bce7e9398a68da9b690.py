"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, Bounds
import itertools


def _compute_radius(centers):
    """Compute the maximum radius for the given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    min_coord = np.min(centers, axis=0)
    max_coord = np.max(centers, axis=0)
    if np.any(min_coord < -0.5) or np.any(max_coord > 0.5):
        return 0.0
    # Distance to boundary
    boundary_dist = np.min([
        np.min(centers[:, 0]) + 0.5,
        0.5 - np.max(centers[:, 0]),
        np.min(centers[:, 1]) + 0.5,
        0.5 - np.max(centers[:, 1])
    ])
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dist, min_pair_dist / 2.0)


def _hexagonal_packing():
    """Generate a hexagonal packing initial configuration."""
    # For 50 circles, use a hexagonal arrangement
    # Rows: try different row counts
    best_config = None
    best_r = 0.0
    
    for n_rows in range(5, 11):
        # Distribute 50 circles among n_rows
        base = N // n_rows
        rem = N % n_rows
        row_counts = [base + (1 if i < rem else 0) for i in range(n_rows)]
        
        centers = []
        for row_idx, count in enumerate(row_counts):
            # Offset alternate rows for hexagonal packing
            offset = 0.0 if row_idx % 2 == 0 else 0.5 / count
            for col_idx in range(count):
                x = -0.5 + (col_idx + 0.5 + offset) / count
                y = -0.5 + (row_idx + 0.5) / n_rows
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    return best_config if best_config is not None else np.zeros((N, 2))


def _random_packing(seed):
    """Generate a random packing with rejection sampling for minimum distance."""
    rng = np.random.RandomState(seed)
    centers = []
    min_dist = 0.15  # Start with a reasonable minimum distance
    
    attempts = 0
    max_attempts = 10000
    
    while len(centers) < N and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(-0.45, 0.45)
        y = rng.uniform(-0.45, 0.45)
        
        valid = True
        for cx, cy in centers:
            if (x - cx)**2 + (y - cy)**2 < min_dist**2:
                valid = False
                break
        
        if valid:
            centers.append([x, y])
    
    if len(centers) < N:
        # Fill remaining with random positions
        while len(centers) < N:
            x = rng.uniform(-0.45, 0.45)
            y = rng.uniform(-0.45, 0.45)
            centers.append([x, y])
    
    return np.array(centers)


def _gradient_descent_optimize(centers, max_iter=5000, lr=0.001, deadline=None):
    """Optimize using gradient descent on the minimum distance objective."""
    centers = centers.copy().astype(np.float64)
    
    def objective_and_grad():
        """Compute negative minimum radius and gradient."""
        global centers
        from scipy.spatial.distance import cdist
        
        # Compute all pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        boundary = np.array([
            centers[:, 0] + 0.5,
            0.5 - centers[:, 0],
            centers[:, 1] + 0.5,
            0.5 - centers[:, 1]
        ])
        boundary_min = np.min(boundary, axis=0)
        
        # Find the minimum
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_pair_idx]
        
        # The radius is min(boundary_min, min_pair_dist/2)
        # We want to maximize this, so minimize negative
        
        # Gradient with respect to each center
        grad = np.zeros_like(centers)
        
        # If boundary is the constraint
        if boundary_min < min_pair_dist / 2:
            # Find which boundary is active
            for i in range(N):
                for k in range(2):
                    if k == 0:
                        d = centers[i, 0] + 0.5
                        d2 = 0.5 - centers[i, 0]
                    else:
                        d = centers[i, 1] + 0.5
                        d2 = 0.5 - centers[i, 1]
                    
                    if d == boundary_min:
                        grad[i, k] += -1.0
                    if d2 == boundary_min:
                        grad[i, k] += 1.0
        else:
            # Pair distance is the constraint
            i, j = min_pair_idx
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 1e-10:
                grad[i, 0] += -dx / dist
                grad[i, 1] += -dy / dist
                grad[j, 0] += dx / dist
                grad[j, 1] += dy / dist
        
        return -min(boundary_min, min_pair_dist / 2), grad
    
    # Simple gradient descent
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = objective_and_grad()
        
        # Clip gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad /= grad_norm
        
        centers -= lr * grad
        
        # Project back to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    return centers


def _simulated_annealing(centers, max_iter=20000, deadline=None):
    """Optimize using simulated annealing."""
    centers = centers.copy().astype(np.float64)
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    step_size = 0.02
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random center to move
        i = rng.randint(N)
        
        # Random perturbation
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        new_centers = centers.copy()
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        
        # Keep in bounds
        new_centers[i, 0] = np.clip(new_centers[i, 0], -0.5, 0.5)
        new_centers[i, 1] = np.clip(new_centers[i, 1], -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better, or with probability exp((new_r - current_r)/T)
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            if rng.random() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
        
        T *= cooling
        step_size *= 0.9995
    
    return best_centers


def construct():
    """Construct the optimal packing of 50 circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Try multiple random restarts with simulated annealing
    rng = np.random.RandomState(123)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            current = centers.copy()
        else:
            current = _random_packing(rng.randint(100000))
        
        # Simulated annealing
        sa_result = _simulated_annealing(current, max_iter=10000, deadline=deadline - 20)
        sa_r = _compute_radius(sa_result)
        
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_result.copy()
        
        # Gradient descent polish
        gd_result = _gradient_descent_optimize(sa_result, max_iter=2000, deadline=deadline - 10)
        gd_r = _compute_radius(gd_result)
        
        if gd_r > best_r:
            best_r = gd_r
            best_centers = gd_result.copy()
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 5:
        final_result = _simulated_annealing(best_centers, max_iter=5000, deadline=deadline - 2)
        final_r = _compute_radius(final_result)
        if final_r > best_r:
            best_centers = final_result
    
    return best_centers
# EVOLVE-BLOCK-END
