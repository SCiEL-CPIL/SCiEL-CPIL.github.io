"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def _check_validity(circles):
    """Check if circles are valid: inside [0,1]^2 and disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0 or x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-7:
                return False
    return True


def _score(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])


def _initial_grid():
    """Create initial grid of circles."""
    # Use a hexagonal-like initial layout for better packing
    # 5x5 grid with some jitter
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 / side if j % 2 == 1 else 0
            x = (i + 0.5) / side + offset * (1.0 / side)
            y = (j + 0.5) / side
            # Clamp to stay in bounds
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            r = 0.5 / side - 0.02
            cells.append([x, y, r])
    return np.array(cells[:N])


def _project_and_grow(circles, max_iter=5000, tol=1e-8):
    """
    Iteratively project circles to satisfy constraints and grow radii.
    Uses a simple iterative approach:
    1. Push overlapping circles apart
    2. Push circles out of boundaries
    3. Grow all radii slightly if feasible
    """
    circles = circles.copy()
    n = len(circles)
    
    # Adaptive radius growth
    growth_rate = 0.001
    min_growth = 1e-6
    max_growth = 0.01
    
    for it in range(max_iter):
        # Project to satisfy constraints
        for _ in range(100):  # Inner iterations for projection
            changed = False
            
            # Pairwise separation
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[j][0] - circles[i][0]
                    dy = circles[j][1] - circles[i][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = circles[i][2] + circles[j][2]
                    
                    if dist < min_dist:
                        if dist < 1e-12:
                            # Random push
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = (min_dist - dist) / 2
                            circles[i][0] -= push * np.cos(angle)
                            circles[i][1] -= push * np.sin(angle)
                            circles[j][0] += push * np.cos(angle)
                            circles[j][1] += push * np.sin(angle)
                        else:
                            push = (min_dist - dist) / 2
                            circles[i][0] -= push * dx / dist
                            circles[i][1] -= push * dy / dist
                            circles[j][0] += push * dx / dist
                            circles[j][1] += push * dy / dist
                        changed = True
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles[i]
                if x - r < 0:
                    circles[i][0] = r
                    changed = True
                if x + r > 1:
                    circles[i][0] = 1 - r
                    changed = True
                if y - r < 0:
                    circles[i][1] = r
                    changed = True
                if y + r > 1:
                    circles[i][1] = 1 - r
                    changed = True
            
            if not changed:
                break
        
        # Check if we can grow radii
        # Try growing all radii by a small amount
        can_grow = True
        for i in range(n):
            x, y, r = circles[i]
            if x - (r + growth_rate) < 0 or x + (r + growth_rate) > 1 or y - (r + growth_rate) < 0 or y + (r + growth_rate) > 1:
                can_grow = False
                break
        
        if can_grow:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[j][0] - circles[i][0]
                    dy = circles[j][1] - circles[i][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = (circles[i][2] + growth_rate) + (circles[j][2] + growth_rate)
                    if dist < min_dist - 1e-9:
                        can_grow = False
                        break
                if not can_grow:
                    break
        
        if can_grow:
            for i in range(n):
                circles[i][2] += growth_rate
            # Keep growth rate
        else:
            # Reduce growth rate
            growth_rate = max(growth_rate * 0.9, min_growth)
        
        if growth_rate < min_growth:
            break
    
    return circles


def _local_optimization(circles, time_limit=10.0):
    """
    Use scipy.optimize.minimize to optimize positions and radii.
    Minimize negative sum of radii subject to constraints.
    """
    start_time = time.time()
    n = len(circles)
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    x0 = circles.copy().flatten()
    
    def objective(v):
        """Negative sum of radii (we minimize, so we maximize sum of radii)."""
        radii = v[2::3]
        return -np.sum(radii)
    
    def gradient(v):
        grad = np.zeros_like(v)
        grad[2::3] = -1.0
        return grad
    
    # Constraints:
    # 1. x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    # 2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    # 3. r_i >= 0
    
    constraints = []
    
    # Boundary constraints (linear)
    # x_i - r_i >= 0
    # -x_i - r_i + 1 >= 0
    # y_i - r_i >= 0
    # -y_i - r_i + 1 >= 0
    
    A_lin = []
    b_lin = []
    
    for i in range(n):
        # x_i - r_i >= 0
        row = np.zeros(3 * n)
        row[3 * i] = 1
        row[3 * i + 2] = -1
        A_lin.append(row)
        b_lin.append(0)
        
        # -x_i - r_i + 1 >= 0
        row = np.zeros(3 * n)
        row[3 * i] = -1
        row[3 * i + 2] = -1
        A_lin.append(row)
        b_lin.append(-1)
        
        # y_i - r_i >= 0
        row = np.zeros(3 * n)
        row[3 * i + 1] = 1
        row[3 * i + 2] = -1
        A_lin.append(row)
        b_lin.append(0)
        
        # -y_i - r_i + 1 >= 0
        row = np.zeros(3 * n)
        row[3 * i + 1] = -1
        row[3 * i + 2] = -1
        A_lin.append(row)
        b_lin.append(-1)
    
    # r_i >= 0
    for i in range(n):
        row = np.zeros(3 * n)
        row[3 * i + 2] = 1
        A_lin.append(row)
        b_lin.append(0)
    
    A_lin = np.array(A_lin)
    b_lin = np.array(b_lin)
    
    # Nonlinear constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    def nonlinear_constraints(v):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = v[3 * i], v[3 * i + 1], v[3 * i + 2]
                xj, yj, rj = v[3 * j], v[3 * j + 1], v[3 * j + 2]
                val = (xi - xj) ** 2 + (yi - yj) ** 2 - (ri + rj) ** 2
                cons.append(val)
        return np.array(cons)
    
    # Use SLSQP with linear constraints and nonlinear constraints
    # For efficiency, we'll use a simpler approach with penalty method
    
    # Actually, let's use a different approach: iterative projection with growth
    # The scipy approach might be too slow for 26 circles with 325 pairwise constraints
    
    return circles  # Return unchanged if we don't have time


def _refine_with_grad_descent(circles, time_limit=20.0):
    """
    Gradient descent on positions and radii with constraint handling.
    """
    start_time = time.time()
    n = len(circles)
    circles = circles.copy()
    
    lr_pos = 0.001
    lr_rad = 0.0001
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros((n, 3))
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[j][0] - circles[i][0]
                dy = circles[j][1] - circles[i][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i][2] + circles[j][2]
                
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx, fy = np.cos(angle), np.sin(angle)
                else:
                    fx, fy = dx / dist, dy / dist
                
                if dist < min_dist:
                    # Push apart
                    strength = (min_dist - dist) * 0.5
                    forces[i][0] -= fx * strength
                    forces[i][1] -= fy * strength
                    forces[j][0] += fx * strength
                    forces[j][1] += fy * strength
        
        # Boundary forces
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                forces[i][0] += (r - x) * 0.5
            if x + r > 1:
                forces[i][0] -= (x + r - 1) * 0.5
            if y - r < 0:
                forces[i][1] += (r - y) * 0.5
            if y + r > 1:
                forces[i][1] -= (y + r - 1) * 0.5
        
        # Update positions
        circles[:, :2] += lr_pos * forces[:, :2]
        
        # Try to grow radii
        # Check if we can grow
        can_grow = True
        for i in range(n):
            x, y, r = circles[i]
            if x - (r + lr_rad) < 0 or x + (r + lr_rad) > 1 or y - (r + lr_rad) < 0 or y + (r + lr_rad) > 1:
                can_grow = False
                break
        
        if can_grow:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[j][0] - circles[i][0]
                    dy = circles[j][1] - circles[i][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = (circles[i][2] + lr_rad) + (circles[j][2] + lr_rad)
                    if dist < min_dist - 1e-9:
                        can_grow = False
                        break
                if not can_grow:
                    break
        
        if can_grow:
            circles[:, 2] += lr_rad
        
        # Clamp to valid region
        for i in range(n):
            x, y, r = circles[i]
            circles[i][0] = np.clip(x, r, 1 - r) if r < 0.5 else 0.5
            circles[i][1] = np.clip(y, r, 1 - r) if r < 0.5 else 0.5
    
    return circles


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns array of shape (26, 3) with rows (x, y, radius).
    """
    deadline = time.time() + 120  # Safety margin
    
    # Start with initial grid
    circles = _initial_grid()
    best_circles = circles.copy()
    best_score = _score(best_circles)
    
    # Phase 1: Projection and growth
    remaining = deadline - time.time()
    if remaining > 10:
        circles = _project_and_grow(circles, max_iter=2000)
        if _check_validity(circles):
            if _score(circles) > best_score:
                best_circles = circles.copy()
                best_score = _score(best_circles)
    
    # Phase 2: Multiple restarts with perturbation
    remaining = deadline - time.time()
    num_restarts = min(5, int(remaining / 15)) if remaining > 15 else 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best
        perturbed = best_circles.copy()
        perturbed[:, :2] += np.random.normal(0, 0.01, (N, 2))
        # Clamp to valid positions
        for i in range(N):
            perturbed[i][0] = np.clip(perturbed[i][0], perturbed[i][2], 1 - perturbed[i][2])
            perturbed[i][1] = np.clip(perturbed[i][1], perturbed[i][2], 1 - perturbed[i][2])
        
        # Project and grow
        refined = _project_and_grow(perturbed, max_iter=1000)
        if _check_validity(refined):
            if _score(refined) > best_score:
                best_circles = refined.copy()
                best_score = _score(refined)
    
    # Phase 3: Fine-tuning with gradient descent
    remaining = deadline - time.time()
    if remaining > 5:
        refined = _refine_with_grad_descent(best_circles, time_limit=min(remaining - 5, 20))
        if _check_validity(refined):
            if _score(refined) > best_score:
                best_circles = refined.copy()
                best_score = _score(refined)
    
    # Final validation
    if not _check_validity(best_circles):
        # Fallback to a valid configuration
        best_circles = _initial_grid()
    
    return best_circles
# EVOLVE-BLOCK-END
