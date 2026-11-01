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
    """Compute smooth penalty for overlaps and boundary violations, plus gradient."""
    n = len(circles)
    penalty = 0.0
    grad = np.zeros_like(circles)
    soft_beta = 50.0  # Smoothness parameter for soft-min
    
    # Boundary penalties (smooth)
    for i in range(n):
        x, y, r = circles[i]
        # Left: violation if x - r < 0
        v = r - x
        if v > -1e-4:
            sv = max(v, 0) + np.log(1 + np.exp(soft_beta * min(v, 0))) / soft_beta
            penalty += 100 * sv * sv
            grad[i, 0] += 200 * sv
            grad[i, 2] += 200 * sv
        # Right: violation if 1 - x - r < 0
        v = r - (1 - x)
        if v > -1e-4:
            sv = max(v, 0) + np.log(1 + np.exp(soft_beta * min(v, 0))) / soft_beta
            penalty += 100 * sv * sv
            grad[i, 0] -= 200 * sv
            grad[i, 2] += 200 * sv
        # Bottom: violation if y - r < 0
        v = r - y
        if v > -1e-4:
            sv = max(v, 0) + np.log(1 + np.exp(soft_beta * min(v, 0))) / soft_beta
            penalty += 100 * sv * sv
            grad[i, 1] += 200 * sv
            grad[i, 2] += 200 * sv
        # Top: violation if 1 - y - r < 0
        v = r - (1 - y)
        if v > -1e-4:
            sv = max(v, 0) + np.log(1 + np.exp(soft_beta * min(v, 0))) / soft_beta
            penalty += 100 * sv * sv
            grad[i, 1] -= 200 * sv
            grad[i, 2] += 200 * sv
    
    # Overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dx = xj - xi
            dy = yj - yi
            dist_sq = dx * dx + dy * dy
            min_dist = ri + rj
            dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
            
            # Violation: min_dist - dist > 0
            v = min_dist - dist
            if v > -1e-4:
                sv = max(v, 0) + np.log(1 + np.exp(soft_beta * min(v, 0))) / soft_beta
                penalty += 100 * sv * sv
                if dist > 1e-10:
                    fx = dx / dist
                    fy = dy / dist
                    grad[i, 0] += 200 * sv * fx
                    grad[i, 1] += 200 * sv * fy
                    grad[i, 2] += 200 * sv
                    grad[j, 0] -= 200 * sv * fx
                    grad[j, 1] -= 200 * sv * fy
                    grad[j, 2] += 200 * sv
                else:
                    grad[i, 0] += 200 * sv
                    grad[j, 0] -= 200 * sv
    
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
    """Gradient descent optimization with adaptive learning rate."""
    circles = circles.copy()
    lr = lr_init
    best_circles = circles.copy()
    best_score = -np.sum(circles[:, 2])
    prev_penalty = np.inf
    stagnation_count = 0
    
    for it in range(max_iter):
        penalty, grad = _compute_penalty_and_grad(circles)
        
        # Objective: maximize sum of radii, penalize violations
        score = np.sum(circles[:, 2]) - penalty
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        if penalty < 1e-10:
            break
        
        # Gradient step: we want to increase radii, so negative gradient on radius
        grad[:, 2] -= 1.0
        
        # Adaptive learning rate based on penalty change
        if penalty < prev_penalty:
            lr = min(lr * 1.05, 0.05)  # Increase LR when making progress
        else:
            lr *= 0.95  # Decrease LR when not making progress
        prev_penalty = penalty
        
        circles -= lr * grad
        
        # Clip to valid range
        for i in range(len(circles)):
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
            circles[i, 2] = np.clip(circles[i, 2], 0.001, 0.5)
        
        # Early termination if stuck
        if stagnation_count > 50:
            break
    
    return best_circles, best_score


def _polish(circles, max_iter=200):
    """Polish a valid packing by iteratively growing radii to their limits."""
    if not _is_valid(circles):
        return circles
    
    circles = circles.copy()
    n = len(circles)
    for it in range(max_iter):
        # Try to grow each circle to its maximum possible radius
        improved = False
        for i in range(n):
            x, y, r = circles[i]
            
            # Boundary limits: max radius given position
            max_r_boundary = min(x, 1 - x, y, 1 - y)
            
            # Other circles limit
            max_r_other = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = circles[j, 0] - x
                dy = circles[j, 1] - y
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-10:
                    # New radius must satisfy: (r_i + r_j) <= dist
                    # => r_i <= dist - r_j
                    available = dist - circles[j, 2]
                    max_r_other = min(max_r_other, available)
            
            max_r = min(max_r_boundary, max_r_other)
            max_r = max(max_r, 0.001)
            
            if max_r > r + 1e-5:
                circles[i, 2] = max_r
                improved = True
        
        if not improved:
            break
    
    return circles


def _local_search(circles, n_trials=50):
    """Local search: coordinate-wise optimization with radius growth."""
    best = circles.copy()
    best_score = np.sum(best[:, 2])
    n = len(best)
    
    for t in range(n_trials):
        trial = best.copy()
        # Pick a random circle to optimize
        i = random.randint(0, n - 1)
        
        # Try small position perturbations
        improved_this_trial = False
        for _ in range(10):
            dx = random.gauss(0, 0.005)
            dy = random.gauss(0, 0.005)
            trial[i, 0] = np.clip(best[i, 0] + dx, 0, 1)
            trial[i, 1] = np.clip(best[i, 1] + dy, 0, 1)
            
            # After moving, try to grow radius
            trial[i, 2] = _max_radius_at(trial, i)
            
            if _is_valid(trial):
                score = np.sum(trial[:, 2])
                if score > best_score + 1e-8:
                    best = trial.copy()
                    best_score = score
                    improved_this_trial = True
                    break
        
        # Also try pure radius growth on current best
        trial2 = best.copy()
        i2 = random.randint(0, n - 1)
        new_r = _max_radius_at(trial2, i2)
        if new_r > trial2[i2, 2] + 1e-6:
            trial2[i2, 2] = new_r
            if _is_valid(trial2):
                score2 = np.sum(trial2[:, 2])
                if score2 > best_score + 1e-8:
                    best = trial2.copy()
                    best_score = score2
    
    return best


def _max_radius_at(circles, i):
    """Compute maximum radius for circle i given its position and other circles."""
    x, y = circles[i, 0], circles[i, 1]
    n = len(circles)
    
    # Boundary limit
    max_r = min(x, 1 - x, y, 1 - y)
    
    # Other circles
    for j in range(n):
        if i == j:
            continue
        dx = circles[j, 0] - x
        dy = circles[j, 1] - y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 1e-10:
            max_r = min(max_r, dist - circles[j, 2])
    
    return max(0.001, max_r)


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
