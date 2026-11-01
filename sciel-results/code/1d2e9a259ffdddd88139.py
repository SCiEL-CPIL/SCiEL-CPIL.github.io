"""Force-directed relaxation for 50 circles in a unit disk."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Compute the largest common radius for given centers."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any pair
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_dist / 2.0, min_boundary)


def _initial_positions_hexagonal():
    """Generate initial positions using a hexagonal lattice pattern."""
    # Start with a hexagonal grid and select N points
    spacing = 0.2
    candidates = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    if len(candidates) >= N:
        # Sort by distance from center and take N
        dists = np.linalg.norm(candidates, axis=1)
        idx = np.argsort(dists)[:N]
        return candidates[idx]
    else:
        # Fallback: random positions
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 0.8, N))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _force_directed_relaxation(points, max_time=120.0):
    """
    Force-directed relaxation with adaptive time-stepping.
    Uses repulsive forces between circles and a harmonic boundary force.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Parameters
    repulsion_strength = 1.0
    boundary_strength = 5.0
    damping = 0.99
    initial_dt = 0.01
    min_dt = 1e-6
    max_dt = 0.05
    
    velocities = np.zeros_like(points)
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    dt = initial_dt
    
    while time.time() - start_time < max_time - 5.0:  # Safety margin
        # Compute forces
        forces = np.zeros_like(points)
        
        # Repulsive forces between circles (spring-like when overlapping)
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Perturb if coincident
                    forces[i] += 0.01 * np.random.randn(2)
                    forces[j] -= 0.01 * np.random.randn(2)
                    continue
                # Repulsive force: stronger when closer
                # Use inverse square law with cutoff
                if dist < 0.5:
                    force_mag = repulsion_strength * (0.5 - dist) / dist
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary force: push circles back if they go outside
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1.0:
                # Harmonic boundary force
                force_mag = boundary_strength * (r - 1.0)
                forces[i] -= force_mag * points[i] / r
            elif r > 0.95:
                # Gentle push inward near boundary
                force_mag = boundary_strength * 0.1 * (r - 0.95)
                forces[i] -= force_mag * points[i] / r
        
        # Adaptive time-stepping: limit max displacement
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            dt = min(max_dt, max_dt * 0.1 / max_force)
            dt = max(dt, min_dt)
        
        # Update velocities with damping
        velocities = damping * velocities + dt * forces
        
        # Update positions
        points += dt * velocities
        
        # Project back inside unit disk if needed
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1.0:
                points[i] = points[i] / r * 1.0
        
        # Compute current radius
        current_radius = _compute_radius(points)
        
        # Keep track of best
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Early stopping if no improvement for a while
        if current_radius < best_radius * 0.999 and dt < min_dt * 10:
            break
    
    return best_points, best_radius


def _gradient_descent_polish(points, max_time=30.0):
    """
    Gradient descent polish using soft-min surrogate for smooth optimization.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Parameters
    lr = 0.001
    decay = 0.999
    soft_min_beta = 50.0  # Higher = sharper min approximation
    
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    while time.time() - start_time < max_time:
        # Compute smooth objective: negative soft-min of all constraints
        # Constraints: d_ij/2 for all i<j, and 1-|c_i| for all i
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                constraints.append(d / 2.0)
            constraints.append(1.0 - np.linalg.norm(points[i]))
        
        constraints = np.array(constraints)
        
        # Soft-min: -log(sum(exp(-beta*c)))/beta
        neg_beta_c = -soft_min_beta * constraints
        # Numerical stability
        max_val = np.max(neg_beta_c)
        log_sum = max_val + np.log(np.sum(np.exp(neg_beta_c - max_val)))
        soft_min_val = -log_sum / soft_min_beta
        
        # Gradient of soft-min with respect to each point
        # d(soft_min)/d(c_k) = -exp(-beta*c_k) / sum(exp(-beta*c_j))
        exp_vals = np.exp(neg_beta_c - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        # Compute gradients for each point
        grads = np.zeros_like(points)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-10:
                    diff = np.array([0.01, 0.01])
                else:
                    diff = points[i] - points[j]
                    diff = diff / d
                w = weights[idx]
                grads[i] += w * diff
                grads[j] -= w * diff
                idx += 1
            # Boundary constraint gradient
            r = np.linalg.norm(points[i])
            if r < 1e-10:
                bgrad = np.array([0.0, 0.0])
            else:
                bgrad = points[i] / r
            w = weights[idx]
            grads[i] -= w * bgrad
            idx += 1
        
        # Update: maximize soft-min, so ascend
        points += lr * grads
        
        # Project inside disk
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1.0:
                points[i] = points[i] / r * 1.0
        
        # Compute actual radius
        current_radius = _compute_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        lr *= decay
        if lr < 1e-8:
            break
    
    return best_points, best_radius


def construct():
    """
    Construct 50 equal circles in a unit disk to maximize common radius.
    """
    start_time = time.time()
    
    # Initial positions
    points = _initial_positions_hexagonal()
    
    # Phase 1: Force-directed relaxation
    points, radius1 = _force_directed_relaxation(points, max_time=60.0)
    
    # Phase 2: Gradient descent polish
    points, radius2 = _gradient_descent_polish(points, max_time=30.0)
    
    # Phase 3: Additional force-directed relaxation from polished state
    points, radius3 = _force_directed_relaxation(points, max_time=20.0)
    
    # Phase 4: Final gradient polish
    points, radius4 = _gradient_descent_polish(points, max_time=15.0)
    
    # Return best result
    # Try multiple restarts if time allows
    best_points = points
    best_radius = max(radius1, radius2, radius3, radius4)
    
    # If we have time, try a few more restarts with different initializations
    remaining_time = 120.0 - (time.time() - start_time)
    if remaining_time > 20.0:
        for restart in range(3):
            # Different initial positions
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + restart * 0.1
            radii = np.sqrt(np.linspace(0.05, 0.85, N))
            init_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            pts, r = _force_directed_relaxation(init_points, max_time=min(25.0, remaining_time/2))
            pts, r = _gradient_descent_polish(pts, max_time=min(15.0, remaining_time/4))
            
            if r > best_radius:
                best_radius = r
                best_points = pts
            
            remaining_time = 120.0 - (time.time() - start_time)
            if remaining_time < 10.0:
                break
    
    # Final validation: ensure all points are inside the disk
    for i in range(len(best_points)):
        r = np.linalg.norm(best_points[i])
        if r > 1.0:
            best_points[i] = best_points[i] / r * 1.0
    
    return best_points

# EVOLVE-BLOCK-END
