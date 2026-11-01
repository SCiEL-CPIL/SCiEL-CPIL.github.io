"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

N = 32

# EVOLVE-BLOCK-START

def _clamp_and_validate(circles, eps=1e-10):
    """Clamp circles to unit square and ensure no overlaps."""
    n = len(circles)
    if n == 0:
        return circles
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2].copy()
    # Clamp centers to [r, 1-r]
    r = np.clip(r, 0, 0.5)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    
    # Iteratively resolve overlaps
    for _ in range(1000):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.hypot(dx, dy)
                min_dist = r[i] + r[j] + eps
                if dist < min_dist:
                    if dist < 1e-12:
                        dx, dy = 1e-8, 1e-8
                        dist = np.hypot(dx, dy)
                    # Push apart
                    needed = min_dist - dist
                    if needed > 0:
                        ux, uy = dx / dist, dy / dist
                        # Move both equally, but respect boundaries
                        push_i = needed * 0.5
                        push_j = needed * 0.5
                        
                        new_xi = x[i] + ux * push_i
                        new_yi = y[i] + uy * push_i
                        new_xj = x[j] - ux * push_j
                        new_yj = y[j] - uy * push_j
                        
                        # Check boundaries
                        if new_xi < r[i] or new_xi > 1.0 - r[i] or new_yi < r[i] or new_yi > 1.0 - r[i]:
                            # Only move j
                            new_xi, new_yi = x[i], y[i]
                            new_xj = x[j] - ux * needed
                            new_yj = y[j] - uy * needed
                        elif new_xj < r[j] or new_xj > 1.0 - r[j] or new_yj < r[j] or new_yj > 1.0 - r[j]:
                            # Only move i
                            new_xi = x[i] + ux * needed
                            new_yi = y[i] + uy * needed
                            new_xj, new_yj = x[j], y[j]
                        
                        x[i], y[i] = new_xi, new_yi
                        x[j], y[j] = new_xj, new_yj
                        improved = True
        if not improved:
            break
    
    return np.column_stack([x, y, r])


def _grid_init(n, ngrid=8):
    """Create a grid of equal circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _greedy_place(circles, positions, radii, n_circles):
    """Greedy placement of circles to maximize sum of radii."""
    n = len(circles)
    # Start with greedy placement
    placed = [False] * n
    
    # Sort by radius descending
    order = np.argsort(-radii)
    
    count = 0
    for idx in order:
        if count >= n_circles:
            break
        x, y, r = positions[idx], positions[idx], radii[idx]
        # Check if this position is valid
        valid = True
        for j in range(n):
            if placed[j]:
                dx = x - positions[j]
                dy = y - positions[j]
                dist = np.hypot(dx, dy)
                if dist < r + radii[j] + 1e-10:
                    valid = False
                    break
        if valid and x >= r and x <= 1.0 - r and y >= r and y <= 1.0 - r:
            circles[count] = [x, y, r]
            placed[idx] = True
            count += 1
    
    return circles, count


def _hill_climb(circles, positions, radii, max_iter=5000, step_size=0.02):
    """Hill climbing optimization to improve packing."""
    n = len(circles)
    current_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()
    best_sum = current_sum
    
    for it in range(max_iter):
        # Pick a random circle
        i = np.random.randint(n)
        # Pick a random direction and magnitude
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0.001, step_size)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_x = circles[i, 0] + dx
        new_y = circles[i, 1] + dy
        r = circles[i, 2]
        
        # Check boundaries
        if new_x < r or new_x > 1.0 - r or new_y < r or new_y > 1.0 - r:
            continue
        
        # Check overlaps
        valid = True
        for j in range(n):
            if j == i:
                continue
            dist = np.hypot(new_x - circles[j, 0], new_y - circles[j, 1])
            if dist < r + circles[j, 2] + 1e-10:
                valid = False
                break
        
        if valid:
            # Try to increase radius
            old_r = circles[i, 2]
            circles[i, 0], circles[i, 1] = new_x, new_y
            
            # Try to expand radius
            max_r = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
            for j in range(n):
                if j == i:
                    continue
                dist = np.hypot(new_x - circles[j, 0], new_y - circles[j, 1])
                max_r = min(max_r, dist - circles[j, 2] - 1e-10)
            max_r = max(max_r, 0)
            
            new_r = max(old_r, max_r)
            circles[i, 2] = new_r
            
            new_sum = np.sum(circles[:, 2])
            if new_sum > best_sum:
                best_sum = new_sum
                best_circles = circles.copy()
            elif new_sum < current_sum:
                # Revert
                circles[i, 0], circles[i, 1], circles[i, 2] = new_x, new_y, old_r
            else:
                current_sum = new_sum
    
    return best_circles


def _simulated_annealing(circles, max_iter=20000, t0=0.05, t1=0.0001, alpha=0.9995):
    """Simulated annealing to improve packing."""
    n = len(circles)
    current = circles.copy()
    best = circles.copy()
    best_sum = np.sum(circles[:, 2])
    current_sum = best_sum
    
    t = t0
    for it in range(max_iter):
        t *= alpha
        
        # Pick a random circle
        i = np.random.randint(n)
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0.001, 0.02)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_x = current[i, 0] + dx
        new_y = current[i, 1] + dy
        r = current[i, 2]
        
        # Check boundaries
        if new_x < r or new_x > 1.0 - r or new_y < r or new_y > 1.0 - r:
            continue
        
        # Check overlaps
        valid = True
        for j in range(n):
            if j == i:
                continue
            dist = np.hypot(new_x - current[j, 0], new_y - current[j, 1])
            if dist < r + current[j, 2] + 1e-10:
                valid = False
                break
        
        if not valid:
            continue
        
        # Try to expand radius
        max_r = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
        for j in range(n):
            if j == i:
                continue
            dist = np.hypot(new_x - current[j, 0], new_y - current[j, 1])
            max_r = min(max_r, dist - current[j, 2] - 1e-10)
        max_r = max(max_r, 0)
        
        new_r = max(r, max_r)
        old_r = current[i, 2]
        current[i, 0], current[i, 1], current[i, 2] = new_x, new_y, new_r
        
        new_sum = np.sum(current[:, 2])
        delta = new_sum - current_sum
        current_sum = new_sum
        
        if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-10)):
            if new_sum > best_sum:
                best_sum = new_sum
                best = current.copy()
        else:
            # Revert
            current[i, 0], current[i, 1], current[i, 2] = new_x, new_y, old_r
            current_sum = current_sum - delta
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial grid
    circles = _grid_init(N)
    positions = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Multiple restarts with different initial configurations
    np.random.seed(42)
    
    # Try different grid sizes
    for ngrid in [4, 5, 6, 7, 8, 9, 10, 12, 16]:
        if time.time() > deadline - 30:
            break
        
        side = int(np.ceil(np.sqrt(N)))
        if ngrid > side:
            continue
            
        cells = []
        for i in range(ngrid):
            for j in range(ngrid):
                cx = (i + 0.5) / ngrid
                cy = (j + 0.5) / ngrid
                r = 0.5 / ngrid - 1e-4
                cells.append((cx, cy, r))
        
        if len(cells) >= N:
            circles = np.array(cells[:N])
            current_sum = np.sum(circles[:, 2])
            
            # Hill climb
            if time.time() < deadline - 60:
                improved = _hill_climb(circles, positions, radii, max_iter=2000, step_size=0.02)
                if np.sum(improved[:, 2]) > best_sum:
                    best_sum = np.sum(improved[:, 2])
                    best_circles = improved.copy()
            
            # Simulated annealing
            if time.time() < deadline - 30:
                improved = _simulated_annealing(circles, max_iter=5000, t0=0.05, t1=0.0001)
                if np.sum(improved[:, 2]) > best_sum:
                    best_sum = np.sum(improved[:, 2])
                    best_circles = improved.copy()
    
    # Final refinement on best
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_circles, max_iter=10000, t0=0.02, t1=0.0001)
        if np.sum(improved[:, 2]) > best_sum:
            best_sum = np.sum(improved[:, 2])
            best_circles = improved.copy()
    
    # Validate and clamp
    best_circles = _clamp_and_validate(best_circles)
    
    return best_circles

# EVOLVE-BLOCK-END
