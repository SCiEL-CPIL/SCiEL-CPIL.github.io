"""Circle packing in unit square: maximize sum of radii for 26 circles."""
import numpy as np
import time

N = 26
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _is_valid(circles, tol=1e-9):
    """Check disjoint interiors and inside unit square."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1.0 + tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - tol:
                return False
    return True


def _objective(circles):
    """Sum of radii (the value to maximize)."""
    return float(np.sum(circles[:, 2]))


def _force_field(circles):
    """Compute gradient of a repulsive energy that pushes circles apart and away from walls.
    
    We maximize sum of radii subject to constraints. Use a Lagrangian-like approach:
    - If two circles overlap, push them apart.
    - If a circle is too close to a wall (x < r, x > 1-r, etc.), push it inward.
    - Also allow radius growth: add a small attractive force to increase radius.
    
    Returns gradient w.r.t. positions (n,2) and radii (n,).
    """
    n = len(circles)
    grad_pos = np.zeros((n, 2))
    grad_r = np.zeros(n)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.hypot(dx, dy)
            min_dist = ri + rj
            if dist < 1e-12:
                # Random push
                angle = np.random.uniform(0, 2 * np.pi)
                dx = np.cos(angle)
                dy = np.sin(angle)
                dist = 1e-12
            overlap = min_dist - dist
            if overlap > 0:
                # Push apart proportional to overlap
                fx = dx / dist * overlap
                fy = dy / dist * overlap
                grad_pos[i, 0] += fx
                grad_pos[i, 1] += fy
                grad_pos[j, 0] -= fx
                grad_pos[j, 1] -= fy
                # Reduce radii to resolve overlap
                grad_r[i] -= 0.5 * overlap
                grad_r[j] -= 0.5 * overlap
    
    # Wall constraints: circle must satisfy x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(n):
        x, y, r = circles[i]
        # Left wall
        if x - r < 0:
            deficit = r - x
            grad_pos[i, 0] += deficit
            grad_r[i] -= deficit
        # Right wall
        if x + r > 1:
            deficit = x + r - 1
            grad_pos[i, 0] -= deficit
            grad_r[i] -= deficit
        # Bottom wall
        if y - r < 0:
            deficit = r - y
            grad_pos[i, 1] += deficit
            grad_r[i] -= deficit
        # Top wall
        if y + r > 1:
            deficit = y + r - 1
            grad_pos[i, 1] -= deficit
            grad_r[i] -= deficit
    
    # Attractive force to grow radii (we want to maximize sum of radii)
    # This is the "objective" gradient
    grad_r[:] += 1.0
    
    return grad_pos, grad_r


def _local_optimize(circles, time_limit, seed):
    """Run gradient ascent with annealing to find a local optimum."""
    np.random.seed(seed)
    n = len(circles)
    circles = circles.copy()
    best_circles = circles.copy()
    best_obj = _objective(best_circles)
    
    # Annealing parameters
    T0 = 0.05
    T_final = 1e-4
    steps = 20000
    learning_rate = 0.01
    
    start_time = time.time()
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
        
        # Cooling schedule
        progress = step / steps
        T = T0 * (T_final / T0) ** progress
        lr = learning_rate * (1.0 - progress * 0.5)
        
        # Compute forces
        grad_pos, grad_r = _force_field(circles)
        
        # Add noise (annealing)
        noise_pos = np.random.normal(0, T, size=(n, 2))
        noise_r = np.random.normal(0, T * 0.1, size=n)
        
        # Update
        new_circles = circles.copy()
        new_circles[:, 0] += lr * (grad_pos[:, 0] + noise_pos[:, 0])
        new_circles[:, 1] += lr * (grad_pos[:, 1] + noise_pos[:, 1])
        new_circles[:, 2] += lr * (grad_r + noise_r)
        
        # Clamp to valid range
        for i in range(n):
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0.0, 1.0)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0.0, 1.0)
            new_circles[i, 2] = max(0.001, new_circles[i, 2])
        
        # Accept if valid and improves, or with probability based on temperature
        new_obj = _objective(new_circles)
        if new_obj > best_obj:
            best_circles = new_circles.copy()
            best_obj = new_obj
        
        # Accept move (Metropolis-like)
        delta = new_obj - _objective(circles)
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-12)):
            circles = new_circles
    
    return best_circles, best_obj


def _initial_guess(seed):
    """Create an initial random configuration."""
    np.random.seed(seed)
    circles = np.zeros((N, 3))
    # Random positions, small initial radii
    circles[:, 0] = np.random.uniform(0.1, 0.9, N)
    circles[:, 1] = np.random.uniform(0.1, 0.9, N)
    circles[:, 2] = 0.02  # Small initial radius
    return circles


def construct():
    """Pack 26 circles in unit square, maximizing sum of radii."""
    start_time = time.time()
    best_circles = None
    best_obj = -np.inf
    
    # Try multiple restarts with different seeds
    seed = 42
    while time.time() - start_time < DEADLINE:
        # Check deadline
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Allocate time for this restart
        restart_time = min(remaining - 2, 10.0)  # Up to 10s per restart
        
        # Initial guess
        circles = _initial_guess(seed)
        
        # Local optimization
        result, obj = _local_optimize(circles, restart_time, seed)
        
        # Validate
        if _is_valid(result):
            if obj > best_obj:
                best_obj = obj
                best_circles = result.copy()
        
        seed += 1
    
    # If no valid result found, return a trivial valid packing
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
