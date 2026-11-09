"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _is_valid(circles, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -eps:
            return False
        if x - r < -eps or x + r > 1 + eps:
            return False
        if y - r < -eps or y + r > 1 + eps:
            return False
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            dist = np.hypot(x - x2, y - y2)
            if dist < r + r2 - eps:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _optimize_single(circles, i, max_iter=500, tol=1e-10):
    """Optimize circle i (position and radius) given others fixed."""
    n = len(circles)
    x0 = circles[i][:3].copy()
    
    def objective(v):
        x, y, r = v
        # Penalty for going outside
        pen = 0.0
        if x - r < 0:
            pen += (x - r) ** 2
        if x + r > 1:
            pen += (x + r - 1) ** 2
        if y - r < 0:
            pen += (y - r) ** 2
        if y + r > 1:
            pen += (y + r - 1) ** 2
        # Penalty for overlap
        for j in range(n):
            if j == i:
                continue
            xj, yj, rj = circles[j]
            dist = np.hypot(x - xj, y - yj)
            min_dist = r + rj
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
        # Maximize radius, so minimize negative radius
        return -r + 100.0 * pen
    
    # Gradient via finite differences would be slow; use L-BFGS-B with bounds
    bounds = [(0, 1), (0, 1), (0, 0.5)]
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': tol})
    
    new_circle = res.x.copy()
    # Ensure validity
    x, y, r = new_circle
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1 or r < 0:
        return circles  # reject invalid
    
    # Check no overlaps
    for j in range(n):
        if j == i:
            continue
        xj, yj, rj = circles[j]
        dist = np.hypot(x - xj, y - yj)
        if dist < r + rj - 1e-9:
            return circles  # reject overlap
    
    new_circles = circles.copy()
    new_circles[i] = new_circle
    return new_circles

def _local_search(circles, max_rounds=20, max_iter_per_circle=200):
    """Iteratively optimize each circle."""
    circles = circles.copy()
    for _ in range(max_rounds):
        improved = False
        for i in range(len(circles)):
            new_circles = _optimize_single(circles, i, max_iter=max_iter_per_circle)
            if _total_radius(new_circles) > _total_radius(circles):
                circles = new_circles
                improved = True
        if not improved:
            break
    return circles

def _random_init(n=26):
    """Generate a random valid initial configuration."""
    circles = []
    for _ in range(n):
        # Try to place a circle with random radius
        for attempt in range(100):
            r = random.uniform(0.02, 0.15)
            x = random.uniform(r, 1 - r)
            y = random.uniform(r, 1 - r)
            # Check against existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = np.hypot(x - cx, y - cy)
                if dist < r + cr - 1e-9:
                    valid = False
                    break
            if valid:
                circles.append([x, y, r])
                break
    # If we couldn't place all, use grid for remaining
    if len(circles) < n:
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                circles.append([x, y, r])
    return np.array(circles[:n])

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    # Start with grid initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_circles = np.array(cells[:N])
    best_score = _total_radius(best_circles)
    
    # Local search from grid
    if time.time() < deadline - 5:
        circles = _local_search(best_circles, max_rounds=10, max_iter_per_circle=100)
        if _is_valid(circles):
            score = _total_radius(circles)
            if score > best_score:
                best_circles = circles
                best_score = score
    
    # Random restarts
    random.seed(42)
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        circles = _random_init(N)
        if len(circles) < N:
            continue
        
        # Local search
        if time.time() < deadline - 10:
            circles = _local_search(circles, max_rounds=15, max_iter_per_circle=150)
        
        if _is_valid(circles):
            score = _total_radius(circles)
            if score > best_score:
                best_circles = circles
                best_score = score
        
        # If we've been stuck for a while, try more aggressive optimization
        if restart_count % 10 == 0 and time.time() < deadline - 20:
            # Try with more rounds
            circles2 = _local_search(best_circles, max_rounds=30, max_iter_per_circle=300)
            if _is_valid(circles2):
                score2 = _total_radius(circles2)
                if score2 > best_score:
                    best_circles = circles2
                    best_score = score2
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        final = _local_search(best_circles, max_rounds=10, max_iter_per_circle=500)
        if _is_valid(final):
            score = _total_radius(final)
            if score > best_score:
                best_circles = final
                best_score = score
    
    return best_circles
# EVOLVE-BLOCK-END
