"""Circle packing in unit square - physics-based gradient descent."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_forces(circles, dt=0.01, repulsion=200.0, attraction=1.0, wall_repulsion=100.0):
    """Compute forces on each circle. Returns (forces, new_radii).
    
    circles: (n, 3) array of (x, y, r)
    """
    n = len(circles)
    forces = np.zeros((n, 2))
    new_radii = circles[:, 2].copy()
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Pairwise repulsion: push apart if overlapping
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            
            if dist < min_dist and dist > 1e-10:
                # Overlapping: strong repulsion
                overlap = min_dist - dist
                # Force magnitude proportional to overlap
                f = repulsion * overlap
                fx = f * dx / dist
                fy = f * dy / dist
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
            elif dist < min_dist * 1.5 and dist > 1e-10:
                # Near overlap: weak attraction to grow radii
                gap = dist - min_dist
                f = attraction * gap
                fx = f * dx / dist
                fy = f * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    
    # Wall repulsion: keep circles inside [0,1]^2
    for i in range(n):
        # Left wall
        if x[i] < r[i]:
            forces[i, 0] += wall_repulsion * (r[i] - x[i])
        # Right wall
        if x[i] + r[i] > 1.0:
            forces[i, 0] -= wall_repulsion * (x[i] + r[i] - 1.0)
        # Bottom wall
        if y[i] < r[i]:
            forces[i, 1] += wall_repulsion * (r[i] - y[i])
        # Top wall
        if y[i] + r[i] > 1.0:
            forces[i, 1] -= wall_repulsion * (y[i] + r[i] - 1.0)
    
    # Try to grow radii if not constrained
    for i in range(n):
        # Check if circle can grow slightly
        can_grow = True
        for j in range(n):
            if i == j:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] + 0.005:
                can_grow = False
                break
        if x[i] - r[i] < -0.005 or x[i] + r[i] > 1.005 or y[i] - r[i] < -0.005 or y[i] + r[i] > 1.005:
            can_grow = False
        
        if can_grow:
            new_radii[i] += 0.002
    
    return forces, new_radii


def _is_valid(circles, tol=1e-6):
    """Check if all circles are disjoint and inside unit square."""
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check inside unit square
    if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
        return False
    
    # Check pairwise disjoint
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _objective(circles):
    """Sum of radii (the quantity to maximize)."""
    return np.sum(circles[:, 2])


def _run_optimization(circles, max_iter=5000, dt=0.01, deadline=None):
    """Run gradient-based optimization with simulated annealing."""
    n = len(circles)
    current = circles.copy()
    best = current.copy()
    best_obj = _objective(best)
    
    # Temperature for simulated annealing
    temp = 0.05
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces, new_radii = _compute_forces(current, dt=dt)
        
        # Update positions
        current[:, 0] += dt * forces[:, 0]
        current[:, 1] += dt * forces[:, 1]
        
        # Update radii (gradually increase)
        current[:, 2] = np.maximum(current[:, 2], new_radii)
        
        # Add small random perturbation (annealing)
        current += np.random.randn(n, 3) * temp
        current[:, 2] = np.maximum(current[:, 2], 0.001)  # Keep radii positive
        
        # Clamp to unit square (soft constraint)
        current[:, 0] = np.clip(current[:, 0], 0.0, 1.0)
        current[:, 1] = np.clip(current[:, 1], 0.0, 1.0)
        
        # Cool down
        temp *= cooling
        
        # Evaluate
        obj = _objective(current)
        
        # Accept if better or with probability based on temperature
        if obj > best_obj or np.random.random() < np.exp(-(best_obj - obj) / max(temp, 1e-8)):
            best = current.copy()
            best_obj = obj
        
        # If we have a valid packing, keep it
        if _is_valid(current, tol=1e-5):
            best = current.copy()
            best_obj = obj
    
    return best, best_obj


def _initial_grid(n):
    """Create initial grid packing."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:n])


def _initial_random(n, seed=42):
    """Create initial random packing with small radii."""
    np.random.seed(seed)
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = np.random.uniform(0.1, 0.9)
        circles[i, 1] = np.random.uniform(0.1, 0.9)
        circles[i, 2] = 0.02
    return circles


def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    best_circles = None
    best_obj = -np.inf
    
    # Multiple restarts with different initializations
    for restart in range(8):
        if time.time() > deadline:
            break
        
        if restart == 0:
            circles = _initial_grid(N)
        elif restart == 1:
            circles = _initial_random(N, seed=restart * 100)
        else:
            # Perturbed versions of previous best
            if best_circles is not None:
                circles = best_circles + np.random.randn(N, 3) * 0.02
                circles[:, 2] = np.maximum(circles[:, 2], 0.01)
            else:
                circles = _initial_random(N, seed=restart * 100)
        
        result, obj = _run_optimization(circles, max_iter=3000, dt=0.005, deadline=deadline)
        
        if obj > best_obj:
            best_obj = obj
            best_circles = result
    
    # Final cleanup: ensure validity
    if best_circles is not None:
        # Try to fix any slight violations
        for _ in range(100):
            if time.time() > deadline:
                break
            if _is_valid(best_circles, tol=1e-6):
                break
            # Shrink slightly and re-optimize
            best_circles[:, 2] *= 0.99
            result, obj = _run_optimization(best_circles, max_iter=500, dt=0.002, deadline=deadline)
            if obj > _objective(best_circles):
                best_circles = result
    
    if best_circles is None:
        # Fallback to grid
        best_circles = _initial_grid(N)
    
    return best_circles
# EVOLVE-BLOCK-END
