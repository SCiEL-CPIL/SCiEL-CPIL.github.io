"""Improved circle packing for n=32 in unit square."""
import numpy as np
import time
import random

N = 32
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _hex_init(n):
    """Hexagonal packing initialization."""
    circles = []
    # Hexagonal grid: rows offset by half
    # For n circles, try to fit in a hex pattern
    # r_hex for hex packing: centers at distance 2r
    # In unit square, max r ~ 0.14 for 32 circles in hex
    r = 0.14
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    row = 0
    while len(circles) < n:
        count_in_row = int(np.floor((1 - 2*r) / dx)) + 1
        offset = (dx / 2) if (row % 2 == 1) else 0
        y = r + row * dy
        if y + r > 1.0:
            break
        for i in range(count_in_row):
            x = r + offset + i * dx
            if x + r <= 1.0 and len(circles) < n:
                circles.append((x, y, r))
        row += 1
    
    # If we don't have enough, add random ones
    while len(circles) < n:
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        circles.append((x, y, r * 0.8))
    
    return np.array(circles[:n])


def _random_init(n):
    """Random initialization."""
    circles = []
    for _ in range(n):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = random.uniform(0.05, 0.15)
        circles.append((x, y, r))
    return np.array(circles[:n])


def _compute_penalty_and_grad(circles):
    """Compute penalty for overlaps and boundary violations, plus gradient."""
    n = len(circles)
    penalty = 0.0
    grad = np.zeros_like(circles)
    
    # Boundary penalties
    for i in range(n):
        x, y, r = circles[i]
        # Left
        if x < r:
            d = r - x
            penalty += 100 * d * d
            grad[i, 0] += 200 * d
            grad[i, 2] += 200 * d
        # Right
        if x > 1 - r:
            d = x - (1 - r)
            penalty += 100 * d * d
            grad[i, 0] -= 200 * d
            grad[i, 2] += 200 * d
        # Bottom
        if y < r:
            d = r - y
            penalty += 100 * d * d
            grad[i, 1] += 200 * d
            grad[i, 2] += 200 * d
        # Top
        if y > 1 - r:
            d = y - (1 - r)
            penalty += 100 * d * d
            grad[i, 1] -= 200 * d
            grad[i, 2] += 200 * d
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dx = xj - xi
            dy = yj - yi
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = ri + rj
            if dist < min_dist:
                d = min_dist - dist
                penalty += 100 * d * d
                if dist > 1e-10:
                    fx = dx / dist
                    fy = dy / dist
                    grad[i, 0] += 200 * d * fx
                    grad[i, 1] += 200 * d * fy
                    grad[i, 2] += 200 * d
                    grad[j, 0] -= 200 * d * fx
                    grad[j, 1] -= 200 * d * fy
                    grad[j, 2] += 200 * d
                else:
                    # Degenerate case
                    grad[i, 0] += 200 * d
                    grad[j, 0] -= 200 * d
    
    return penalty, grad


def _is_valid(circles, tol=1e-6):
    """Check if all circles are valid (inside square, no overlaps)."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x < r - tol or x > 1 - r + tol or y < r - tol or y > 1 - r + tol:
            return False
        if r <= 0:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[j, 0] - circles[i, 0]
            dy = circles[j, 1] - circles[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True


def _optimize(circles, max_iter=500, lr_init=0.01, lr_decay=0.999):
    """Gradient descent optimization."""
    circles = circles.copy()
    lr = lr_init
    best_circles = circles.copy()
    best_score = -np.sum(circles[:, 2])
    
    for it in range(max_iter):
        penalty, grad = _compute_penalty_and_grad(circles)
        
        # Objective: maximize sum of radii, penalize violations
        score = np.sum(circles[:, 2]) - penalty
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        if penalty < 1e-8:
            break
        
        # Gradient step
        grad[:, 2] -= 1.0  # We want to increase radii
        circles -= lr * grad
        
        # Clip to valid range
        for i in range(len(circles)):
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
            circles[i, 2] = np.clip(circles[i, 2], 0.001, 0.5)
        
        lr *= lr_decay
    
    return best_circles, best_score


def _polish(circles, max_iter=200):
    """Polish a valid packing by growing radii."""
    if not _is_valid(circles):
        return circles
    
    circles = circles.copy()
    for it in range(max_iter):
        # Try to grow each circle slightly
        improved = False
        for i in range(len(circles)):
            # Find max growth possible
            max_growth = 0.001
            x, y, r = circles[i]
            
            # Boundary limits
            max_growth = min(max_growth, x - r, 1 - x - r, y - r, 1 - y - r)
            
            # Other circles
            for j in range(len(circles)):
                if i == j:
                    continue
                dx = circles[j, 0] - x
                dy = circles[j, 1] - y
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-10:
                    available = dist - r - circles[j, 2]
                    max_growth = min(max_growth, available * 0.5)
            
            if max_growth > 0.0001:
                new_r = r + max_growth * 0.5
                circles[i, 2] = new_r
                improved = True
        
        if not improved:
            break
    
    return circles


def _local_search(circles, n_trials=50):
    """Local search: try small perturbations."""
    best = circles.copy()
    best_score = np.sum(best[:, 2])
    
    for t in range(n_trials):
        trial = best.copy()
        # Perturb a random circle
        i = random.randint(0, len(trial) - 1)
        trial[i, 0] += random.gauss(0, 0.01)
        trial[i, 1] += random.gauss(0, 0.01)
        trial[i, 2] *= (1 + random.gauss(0, 0.05))
        trial[i, 0] = np.clip(trial[i, 0], 0, 1)
        trial[i, 1] = np.clip(trial[i, 1], 0, 1)
        trial[i, 2] = np.clip(trial[i, 2], 0.001, 0.5)
        
        if _is_valid(trial):
            score = np.sum(trial[:, 2])
            if score > best_score:
                best = trial.copy()
                best_score = score
    
    return best


def construct():
    start_time = time.time()
    
    # Multiple restart strategies
    best_circles = None
    best_score = -np.inf
    
    # Start with hex initialization
    inits = [_hex_init(N)]
    
    # Add some random inits
    for _ in range(3):
        inits.append(_random_init(N))
    
    for init in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        # Optimize
        circles, score = _optimize(init, max_iter=300, lr_init=0.01, lr_decay=0.999)
        
        # Check validity
        if _is_valid(circles):
            # Polish
            circles = _polish(circles, max_iter=100)
            score = np.sum(circles[:, 2])
            
            # Local search
            if time.time() - start_time < DEADLINE - 10:
                circles = _local_search(circles, n_trials=30)
                score = np.sum(circles[:, 2])
            
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    # Final polish on best
    if best_circles is not None and _is_valid(best_circles):
        best_circles = _polish(best_circles, max_iter=50)
    
    if best_circles is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles

# EVOLVE-BLOCK-END
