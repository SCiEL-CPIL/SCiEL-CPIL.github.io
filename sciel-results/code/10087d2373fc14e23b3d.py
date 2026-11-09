"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x + r > 1 + eps) or np.any(x - r < -eps) or np.any(y + r > 1 + eps) or np.any(y - r < -eps):
        return False
    if np.any(r < -eps):
        return False
    
    # Check non-overlapping
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True


def _objective(circles):
    """Objective: maximize sum of radii, with penalty for violations."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(circles)
    
    reward = np.sum(r)
    
    penalty = 0.0
    penalty += np.sum(np.maximum(0, x + r - 1)) ** 2
    penalty += np.sum(np.maximum(0, -x + r)) ** 2
    penalty += np.sum(np.maximum(0, y + r - 1)) ** 2
    penalty += np.sum(np.maximum(0, -y + r)) ** 2
    
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_penalty += overlap ** 2
    
    return reward - 50 * penalty - 50 * overlap_penalty


def _gradient_descent(circles, max_iter=20000, lr=0.005, deadline=None):
    """Improve packing using gradient ascent on the objective."""
    circles = circles.copy()
    n = len(circles)
    
    best_circles = circles.copy()
    best_score = _objective(circles)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        
        grad_r += 1.0
        
        for i in range(n):
            if x[i] + r[i] > 1:
                grad_x[i] += 100
                grad_r[i] -= 100
            if x[i] - r[i] < 0:
                grad_x[i] -= 100
                grad_r[i] -= 100
            if y[i] + r[i] > 1:
                grad_y[i] += 100
                grad_r[i] -= 100
            if y[i] - r[i] < 0:
                grad_y[i] -= 100
                grad_r[i] -= 100
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    if dist > 1e-10:
                        ux = dx / dist
                        uy = dy / dist
                        grad_x[i] -= 100 * overlap * ux
                        grad_y[i] -= 100 * overlap * uy
                        grad_x[j] += 100 * overlap * ux
                        grad_y[j] += 100 * overlap * uy
                        grad_r[i] -= 100 * overlap
                        grad_r[j] -= 100 * overlap
                    else:
                        grad_x[i] += 100
                        grad_x[j] -= 100
        
        circles[:, 0] += lr * grad_x
        circles[:, 1] += lr * grad_y
        circles[:, 2] += lr * grad_r * 0.1
        
        circles[:, 0] = np.clip(circles[:, 0], 0, 1)
        circles[:, 1] = np.clip(circles[:, 1], 0, 1)
        circles[:, 2] = np.maximum(circles[:, 2], 0.001)
        
        score = _objective(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    return best_circles


def _simulated_annealing(circles, max_iter=5000, deadline=None):
    """Simulated annealing for better exploration."""
    circles = circles.copy()
    n = len(circles)
    current_score = _objective(circles)
    best_score = current_score
    best_circles = circles.copy()
    
    T = 0.01
    alpha = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        new_circles = circles.copy()
        
        # Random perturbation
        idx = random.randint(0, n - 1)
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            new_circles[idx, 0] += random.gauss(0, 0.03)
            new_circles[idx, 1] += random.gauss(0, 0.03)
        elif move_type == 1:
            new_circles[idx, 2] *= random.uniform(0.95, 1.05)
        else:
            i, j = random.sample(range(n), 2)
            new_circles[[i, j], :] = new_circles[[j, i], :]
        
        new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
        new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 0.001)
        
        new_score = _objective(new_circles)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(T, 1e-9)):
            circles = new_circles
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_circles = circles.copy()
        
        T *= alpha
    
    return best_circles


def _construct_grid():
    """Create a grid-based initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
            count += 1
    return np.array(cells)


def _construct_random():
    """Create a random initialization."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.02, 0.06, N)
    return np.column_stack([x, y, r])


def construct():
    """Construct optimal circle packing in unit square."""
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    best_circles = None
    best_score = -np.inf
    
    np.random.seed(42)
    random.seed(42)
    
    # Try multiple strategies
    strategies = []
    
    # Grid-based
    for _ in range(3):
        grid = _construct_grid()
        strategies.append(('grid', grid))
    
    # Random
    for _ in range(5):
        rand = _construct_random()
        strategies.append(('random', rand))
    
    # Mixed: some grid, some random
    mixed = _construct_grid()
    idx = np.random.choice(N, 5, replace=False)
    mixed[idx, 0] = np.random.uniform(0.1, 0.9, 5)
    mixed[idx, 1] = np.random.uniform(0.1, 0.9, 5)
    strategies.append(('mixed', mixed))
    
    for name, initial in strategies:
        if time.time() > deadline - 20:
            break
        
        # Gradient descent
        gd_result = _gradient_descent(initial.copy(), max_iter=5000, lr=0.005, deadline=deadline - 15)
        gd_score = _objective(gd_result)
        
        if gd_score > best_score:
            best_score = gd_score
            best_circles = gd_result.copy()
        
        # Simulated annealing on the GD result
        if time.time() > deadline - 10:
            break
        sa_result = _simulated_annealing(gd_result.copy(), max_iter=2000, deadline=deadline - 10)
        sa_score = _objective(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_circles = sa_result.copy()
    
    # Final validation
    if best_circles is None or not _is_valid(best_circles):
        # Fallback to a valid configuration
        side = 6
        circles = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                circles.append([x, y, r])
                count += 1
        best_circles = np.array(circles)
    
    return best_circles
# EVOLVE-BLOCK-END
