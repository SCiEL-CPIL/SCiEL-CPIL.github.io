"""Improved packing for 50 equal circles in a unit square [-1/2, 1/2]^2."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraint: distance from each center to nearest wall
    margins = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    wall_r = np.min(margins)
    if wall_r < 0:
        return 0.0
    
    # Pairwise distances
    d = pdist(centers)
    min_pair_dist = np.min(d) if len(d) > 0 else np.inf
    
    r = min(min_pair_dist / 2.0, wall_r)
    return r


def _objective_and_grad(x):
    """Objective: minimize -radius. Returns (value, gradient)."""
    centers = x.reshape(N, 2)
    r = _compute_radius(centers)
    
    # Compute gradient via finite differences (approximate)
    # For efficiency, we'll use a simple approach
    grad = np.zeros_like(x)
    eps = 1e-5
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        r_plus = _compute_radius(x_plus.reshape(N, 2))
        x_minus = x.copy()
        x_minus[i] -= eps
        r_minus = _compute_radius(x_minus.reshape(N, 2))
        grad[i] = -(r_plus - r_minus) / (2 * eps)
    
    return -r, grad


def _force_directed(centers, steps=5000, dt=0.001, k_repel=1.0, k_wall=2.0):
    """Force-directed relaxation to improve packing."""
    centers = centers.copy()
    n = len(centers)
    
    for step in range(steps):
        forces = np.zeros_like(centers)
        
        # Repulsion between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    # Random small force if coincident
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                else:
                    # Inverse square repulsion
                    force_mag = k_repel / (dist ** 2 + 1e-6)
                    force_dir = diff / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        
        # Wall repulsion: push centers away from walls
        for i in range(n):
            for axis in range(2):
                val = centers[i, axis]
                if val > 0:
                    dist_to_wall = 0.5 - val
                    if dist_to_wall < 0.1:
                        forces[i, axis] -= k_wall / (dist_to_wall + 1e-6)
                else:
                    dist_to_wall = 0.5 + val
                    if dist_to_wall < 0.1:
                        forces[i, axis] += k_wall / (dist_to_wall + 1e-6)
        
        # Update positions
        centers += dt * forces
        
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay dt
        dt *= 0.999
    
    return centers


def _gradient_refine(centers, max_iter=1000, tol=1e-8):
    """Gradient-based refinement using L-BFGS-B."""
    x = centers.flatten()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        centers = x.reshape(N, 2)
        return -_compute_radius(centers)
    
    result = minimize(obj, x, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    
    return result.x.reshape(N, 2), -result.fun


def _initial_hex(n):
    """Create a hexagonal lattice initialization for n circles in the unit square."""
    # Create hexagonal lattice points
    dx = 0.14  # horizontal spacing
    dy = dx * np.sqrt(3) / 2  # vertical spacing for hex
    
    # Generate points in a grid larger than the square
    points = []
    for i in range(20):
        y = i * dy - 0.5
        for j in range(20):
            x = j * dx - 0.5 + (0.5 * dx if i % 2 else 0)
            points.append([x, y])
    
    points = np.array(points)
    # Filter to those inside the square with some margin
    mask = (points[:, 0] >= -0.45) & (points[:, 0] <= 0.45) & (points[:, 1] >= -0.45) & (points[:, 1] <= 0.45)
    points = points[mask]
    
    # If we have more than n, randomly sample n
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, (n - len(points), 2))
        points = np.vstack([points, extra])
    
    return points


def _simulated_annealing(centers, max_iter=100000, T_start=0.05, T_end=1e-7, cooling=0.99995, deadline=None):
    """Simulated annealing to optimize circle packing."""
    n = len(centers)
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    current_r = best_r
    T = T_start
    
    for it in range(max_iter):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random circle to move
        i = np.random.randint(n)
        # Random displacement scaled with temperature
        scale = T * 5
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        new_x = centers[i, 0] + dx
        new_y = centers[i, 1] + dy
        
        # Clamp to square
        new_x = np.clip(new_x, -0.5, 0.5)
        new_y = np.clip(new_y, -0.5, 0.5)
        
        old_x, old_y = centers[i, 0], centers[i, 1]
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        new_r = _compute_radius(centers)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta = new_r - current_r
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() > prob:
                    centers[i, 0] = old_x
                    centers[i, 1] = old_y
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    return best_centers, best_r


def _local_refinement(centers, max_iter=20000):
    """Local refinement using coordinate descent with adaptive step sizes."""
    n = len(centers)
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    step_size = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for dx in [0, -step_size, step_size]:
                for dy in [0, -step_size, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    old_x, old_y = centers[i, 0], centers[i, 1]
                    new_x = np.clip(old_x + dx, -0.5, 0.5)
                    new_y = np.clip(old_y + dy, -0.5, 0.5)
                    centers[i, 0] = new_x
                    centers[i, 1] = new_y
                    new_r = _compute_radius(centers)
                    if new_r > current_r:
                        current_r = new_r
                        improved = True
                    else:
                        centers[i, 0] = old_x
                        centers[i, 1] = old_y
        if not improved:
            step_size *= 0.5
            no_improve += 1
            if no_improve > 8 or step_size < 1e-7:
                break
        else:
            no_improve = 0
    
    return centers, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s budget
    
    best_centers = None
    best_r = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initial hexagonal configuration
        if restart == 0:
            centers = _initial_hex(N)
        else:
            # Perturb the best found so far
            if best_centers is not None:
                centers = best_centers + np.random.randn(N, 2) * 0.05
                centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
                centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            else:
                centers = _initial_hex(N)
        
        # Force-directed relaxation
        fd_centers = _force_directed(centers, steps=2000)
        fd_r = _compute_radius(fd_centers)
        
        if fd_r > best_r:
            best_r = fd_r
            best_centers = fd_centers
        
        # Simulated annealing
        remaining_time = deadline - time.time() - 15  # reserve 15s for refinement
        if remaining_time > 5:
            max_iter = int(remaining_time * 5000)  # estimate iterations
            sa_centers, sa_r = _simulated_annealing(
                fd_centers,
                max_iter=max_iter,
                T_start=0.03,
                T_end=1e-7,
                cooling=0.99997,
                deadline=deadline - 15
            )
            
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers
        
        # Gradient-based refinement
        if time.time() < deadline - 10:
            grad_centers, grad_r = _gradient_refine(best_centers, max_iter=500)
            
            if grad_r > best_r:
                best_r = grad_r
                best_centers = grad_centers
        
        # Local refinement with adaptive steps
        if time.time() < deadline - 5:
            ref_centers, ref_r = _local_refinement(best_centers, max_iter=10000)
            
            if ref_r > best_r:
                best_r = ref_r
                best_centers = ref_centers
    
    return best_centers
# EVOLVE-BLOCK-END
