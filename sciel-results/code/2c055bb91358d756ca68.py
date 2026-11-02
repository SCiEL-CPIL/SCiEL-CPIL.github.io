"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _check_validity(circles):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _penalty(circles, penalty_weight=1e4):
    """Compute penalty for invalid configuration."""
    n = len(circles)
    pen = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Boundary violations
        if x - r < 0:
            pen += (x - r) ** 2
        if x + r > 1:
            pen += (x + r - 1) ** 2
        if y - r < 0:
            pen += (y - r) ** 2
        if y + r > 1:
            pen += (y + r - 1) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = circles[i][2] + circles[j][2]
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return penalty_weight * pen


def _objective(params, penalty_weight=1e4):
    """Objective: minimize negative sum of radii + penalty."""
    n = len(params) // 3
    circles = params.reshape(n, 3)
    # Ensure radii are positive
    radii = np.maximum(circles[:, 2], 1e-6)
    penalty = _penalty(circles, penalty_weight)
    return -np.sum(radii) + penalty


def _hex_initialization(n_circles):
    """Generate hexagonal packing initialization."""
    # Start with a hexagonal grid pattern
    r0 = 0.15
    dx = 2 * r0
    dy = 2 * r0 * np.sqrt(3) / 2
    circles = []
    for row in range(8):
        for col in range(8):
            x = col * dx + (row % 2) * dx / 2 + 0.1
            y = row * dy + 0.1
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles.append([x, y, r0])
    # If we have more than needed, take the first n
    if len(circles) > n_circles:
        circles = circles[:n_circles]
    # If we have fewer, add random circles
    while len(circles) < n_circles:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        circles.append([x, y, r0])
    return np.array(circles)


def _grid_initialization(n_circles):
    """Generate grid initialization."""
    side = int(np.ceil(np.sqrt(n_circles)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) < n_circles:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                circles.append([x, y, r])
    return np.array(circles)


def _corner_weighted_initialization(n_circles):
    """Initialize with circles weighted towards corners and edges."""
    circles = []
    # Place circles near corners
    corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
                        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    for x, y in corner_positions:
        if len(circles) < n_circles:
            circles.append([x + np.random.uniform(-0.05, 0.05),
                            y + np.random.uniform(-0.05, 0.05),
                            0.12])
    # Fill rest with random
    while len(circles) < n_circles:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        circles.append([x, y, 0.1])
    return np.array(circles)


def _local_optimize(circles, max_iter=500, tol=1e-8):
    """Local optimization using gradient descent with penalty method."""
    n = len(circles)
    params = circles.flatten().copy()
    
    best_params = params.copy()
    best_obj = _objective(params)
    
    lr = 0.01
    for it in range(max_iter):
        # Compute gradient numerically
        grad = np.zeros_like(params)
        for i in range(len(params)):
            eps = 1e-5
            params_plus = params.copy()
            params_plus[i] += eps
            obj_plus = _objective(params_plus)
            params[i] = params_plus[i]
            grad[i] = (obj_plus - best_obj) / eps
            best_obj = _objective(params)
        
        # Gradient step
        params -= lr * grad
        
        # Project: ensure radii are positive and positions in [0,1]
        for k in range(n):
            params[3*k] = np.clip(params[3*k], 0, 1)
            params[3*k+1] = np.clip(params[3*k+1], 0, 1)
            params[3*k+2] = np.maximum(params[3*k+2], 1e-6)
        
        obj = _objective(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.9
        
        if abs(grad).max() < tol:
            break
    
    return best_params.reshape(n, 3)


def _delaunay_contact_graph(circles):
    """Build contact graph using Delaunay triangulation."""
    points = circles[:, :2]
    try:
        tri = Delaunay(points)
        # Get edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return []


def _anneal_optimize(circles, start_time, deadline, n_replicas=4, max_steps=2000):
    """Parallel tempering Monte Carlo optimization."""
    n = len(circles)
    # Create replicas at different temperatures
    temps = [0.01, 0.05, 0.1, 0.5]
    replicas = [circles.copy() for _ in range(n_replicas)]
    
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    step_size = 0.02
    
    for step in range(max_steps):
        if time.time() - start_time > deadline - 10:
            break
        
        for rep in range(n_replicas):
            # Propose move: perturb one circle
            idx = np.random.randint(n)
            move_type = np.random.choice(3)  # 0: position, 1: radius, 2: both
            
            new_circles = replicas[rep].copy()
            if move_type == 0:
                new_circles[idx, 0] += np.random.normal(0, step_size)
                new_circles[idx, 1] += np.random.normal(0, step_size)
            elif move_type == 1:
                new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.5))
                new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.5)
            else:
                new_circles[idx, 0] += np.random.normal(0, step_size)
                new_circles[idx, 1] += np.random.normal(0, step_size)
                new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.3))
                new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.5)
            
            # Ensure positions in [0,1]
            new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
            
            # Compute energy change (penalty-based)
            old_energy = _penalty(replicas[rep], 100) - np.sum(replicas[rep][:, 2])
            new_energy = _penalty(new_circles, 100) - np.sum(new_circles[:, 2])
            delta_e = new_energy - old_energy
            
            # Metropolis acceptance
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temps[rep]):
                replicas[rep] = new_circles
            
            # Track best
            current_sum = np.sum(replicas[rep][:, 2])
            if _check_validity(replicas[rep]) and current_sum > best_sum:
                best_sum = current_sum
                best_circles = replicas[rep].copy()
        
        # Swap replicas (Metropolis swap)
        for rep in range(n_replicas - 1):
            if np.random.random() < 0.3:  # Swap with some probability
                # Compute swap acceptance
                e1 = _penalty(replicas[rep], 100) - np.sum(replicas[rep][:, 2])
                e2 = _penalty(replicas[rep + 1], 100) - np.sum(replicas[rep + 1][:, 2])
                delta_swap = (temps[rep] - temps[rep + 1]) * (e1 - e2)
                if delta_swap < 0 or np.random.random() < np.exp(-delta_swap):
                    replicas[rep], replicas[rep + 1] = replicas[rep + 1], replicas[rep]
        
        # Adaptive step size
        if step % 100 == 0:
            step_size *= 0.95
    
    return best_circles


def construct():
    start_time = time.time()
    
    # Multiple initialization strategies
    initializations = [
        _hex_initialization(N),
        _grid_initialization(N),
        _corner_weighted_initialization(N)
    ]
    
    best_circles = None
    best_sum = 0.0
    
    for init in initializations:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Local optimization
        optimized = _local_optimize(init, max_iter=200)
        
        # Check validity and sum
        if _check_validity(optimized):
            s = np.sum(optimized[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = optimized.copy()
    
    # Parallel tempering refinement
    if best_circles is not None and time.time() - start_time < DEADLINE - 20:
        refined = _anneal_optimize(best_circles, start_time, DEADLINE)
        if _check_validity(refined):
            s = np.sum(refined[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = refined.copy()
    
    # Final polish: high-precision local optimization
    if best_circles is not None and time.time() - start_time < DEADLINE - 5:
        polished = _local_optimize(best_circles, max_iter=1000, tol=1e-10)
        if _check_validity(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_circles = polished.copy()
                best_sum = s
    
    if best_circles is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
