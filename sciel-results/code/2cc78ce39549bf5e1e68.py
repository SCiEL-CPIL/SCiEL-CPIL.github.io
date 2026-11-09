"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
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
    
    # Reward: sum of radii
    reward = np.sum(r)
    
    # Penalty for going outside bounds
    penalty = 0.0
    penalty += np.sum(np.maximum(0, r - x)) ** 2
    penalty += np.sum(np.maximum(0, r - (1 - x))) ** 2
    penalty += np.sum(np.maximum(0, r - y)) ** 2
    penalty += np.sum(np.maximum(0, r - (1 - y))) ** 2
    
    # Penalty for overlapping
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_penalty += overlap ** 2
    
    return reward - 10 * penalty - 10 * overlap_penalty


def _gradient_descent(circles, max_iter=5000, lr=0.01, deadline=None):
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
        
        # Compute gradients
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        
        # Reward gradient: maximize sum of radii
        grad_r += 1.0
        
        # Boundary penalty gradients
        for i in range(n):
            # Left boundary
            if r[i] > x[i]:
                grad_x[i] -= 2 * (r[i] - x[i])
                grad_r[i] -= 2 * (r[i] - x[i])
            # Right boundary
            if r[i] > 1 - x[i]:
                grad_x[i] += 2 * (r[i] - (1 - x[i]))
                grad_r[i] -= 2 * (r[i] - (1 - x[i]))
            # Bottom boundary
            if r[i] > y[i]:
                grad_y[i] -= 2 * (r[i] - y[i])
                grad_r[i] -= 2 * (r[i] - y[i])
            # Top boundary
            if r[i] > 1 - y[i]:
                grad_y[i] += 2 * (r[i] - (1 - y[i]))
                grad_r[i] -= 2 * (r[i] - (1 - y[i]))
        
        # Overlap penalty gradients
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
                        # Push circles apart
                        grad_x[i] -= 2 * overlap * ux
                        grad_y[i] -= 2 * overlap * uy
                        grad_x[j] += 2 * overlap * ux
                        grad_y[j] += 2 * overlap * uy
                        grad_r[i] -= 2 * overlap
                        grad_r[j] -= 2 * overlap
        
        # Apply gradient ascent
        circles[:, 0] += lr * grad_x
        circles[:, 1] += lr * grad_r * 0.1  # Scale radius updates
        circles[:, 2] += lr * grad_r
        
        # Keep circles inside bounds and positive radius
        circles[:, 0] = np.clip(circles[:, 0], 0, 1)
        circles[:, 1] = np.clip(circles[:, 1], 0, 1)
        circles[:, 2] = np.maximum(circles[:, 2], 0.001)
        
        score = _objective(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    return best_circles


def construct():
    """Construct optimal circle packing in unit square."""
    deadline = time.time() + 120  # Safety margin
    
    # Start with a grid initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    circles = np.array(cells[:N])
    
    # Try multiple random restarts
    best_circles = circles.copy()
    best_score = _objective(circles)
    
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        if restart == 0:
            current = circles.copy()
        else:
            # Random positions with small radii
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.02, 0.08, N)
            current = np.column_stack([x, y, r])
        
        # Optimize
        optimized = _gradient_descent(current, max_iter=3000, lr=0.02, deadline=deadline - 10)
        score = _objective(optimized)
        
        if score > best_score:
            best_score = score
            best_circles = optimized
    
    return best_circles
# EVOLVE-BLOCK-END
