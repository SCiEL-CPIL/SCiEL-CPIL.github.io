"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(circles, eps=1e-9):
    """Check all constraints: inside unit square, disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -eps or y - r < -eps or x + r > 1 + eps or y + r > 1 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            if dist_sq < (ri + rj)**2 - eps:
                return False
    return True

def _objective(circles):
    """Sum of radii (the score we maximize)."""
    return float(sum(c[2] for c in circles))

def _hex_init(n=32):
    """Generate a hexagonal lattice initial placement."""
    # Approximate hex packing
    # 6 circles per row for 5 rows = 30, plus 2 more?
    # Let's use a rectangular grid with hex offset.
    cols = 6
    rows = int(np.ceil(n / cols))
    cells = []
    dx = 1.0 / cols
    dy = 1.0 / rows * 0.866 # Hex vertical spacing
    # Adjust to fit in [0,1]
    # We want to center the grid.
    
    # Simpler: Just use a dense grid with hex offset
    for i in range(rows):
        for j in range(cols):
            if len(cells) >= n:
                break
            # Hex offset: shift odd rows by half a step
            offset = 0.5 * dx if i % 2 == 1 else 0.0
            x = (j + 0.5 + offset) * dx
            y = (i + 0.5) * dy
            
            # Clamp to ensure we stay in bounds, adjusting radius
            # Initial radius estimate
            r = min(dx, dy) * 0.45
            
            # Ensure x,y,r are within [0,1]
            x = max(r, min(1 - r, x))
            y = max(r, min(1 - r, y))
            
            cells.append([x, y, r])
    return cells

def _random_init(n=32, seed=None):
    """Generate a random initial placement with small circles."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    circles = []
    for _ in range(n):
        r = random.uniform(0.01, 0.08)
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        circles.append([x, y, r])
    return circles

def _local_search(circles, max_iters=5000, lr=0.01, deadline=None):
    """
    Local search: iteratively perturb positions and radii, accept if valid and improves objective.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    best_obj = _objective(circles)
    best_circles = [c[:] for c in circles]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random circle to perturb
        i = random.randint(0, n - 1)
        
        # Perturb position
        dx = random.gauss(0, lr * 0.5)
        dy = random.gauss(0, lr * 0.5)
        # Perturb radius
        dr = random.gauss(0, lr * 0.3)
        
        new_circles = [c[:] for c in circles]
        new_circles[i][0] += dx
        new_circles[i][1] += dy
        new_circles[i][2] = max(0.001, new_circles[i][2] + dr)
        
        # Check if still in bounds
        x, y, r = new_circles[i]
        if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
            continue
        
        # Check validity
        if _is_valid(new_circles):
            new_obj = _objective(new_circles)
            if new_obj > best_obj:
                circles = new_circles
                best_obj = new_obj
                best_circles = [c[:] for c in circles]
    
    return best_circles, best_obj

def _hill_climb_radii(circles, max_iters=2000, deadline=None):
    """
    Try to increase radii while maintaining validity.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    best_obj = _objective(circles)
    best_circles = [c[:] for c in circles]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = random.randint(0, n - 1)
        dr = random.uniform(0.001, 0.01)
        
        new_circles = [c[:] for c in circles]
        new_circles[i][2] += dr
        
        x, y, r = new_circles[i]
        if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
            continue
        
        if _is_valid(new_circles):
            new_obj = _objective(new_circles)
            if new_obj > best_obj:
                circles = new_circles
                best_obj = new_obj
                best_circles = [c[:] for c in circles]
    
    return best_circles, best_obj

def _pairwise_separation(circles, max_iters=3000, deadline=None):
    """
    For each pair that is too close, push them apart.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    best_obj = _objective(circles)
    best_circles = [c[:] for c in circles]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find a violating pair
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        xi, yi, ri = circles[i]
        xj, yj, rj = circles[j]
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        min_dist = ri + rj
        
        if dist < min_dist:
            # Push apart
            if dist < 1e-12:
                dx = random.gauss(0, 0.01)
                dy = random.gauss(0, 0.01)
            else:
                dx = (xi - xj) / dist
                dy = (yi - yj) / dist
            
            push = (min_dist - dist) / 2 + 0.001
            new_circles = [c[:] for c in circles]
            new_circles[i][0] += dx * push
            new_circles[i][1] += dy * push
            new_circles[j][0] -= dx * push
            new_circles[j][1] -= dy * push
            
            # Check bounds
            valid = True
            for k in range(n):
                x, y, r = new_circles[k]
                if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                    valid = False
                    break
            
            if valid and _is_valid(new_circles):
                new_obj = _objective(new_circles)
                if new_obj > best_obj:
                    circles = new_circles
                    best_obj = new_obj
                    best_circles = [c[:] for c in circles]
    
    return best_circles, best_obj

def construct():
    """
    Main entry point: pack 32 circles in unit square maximizing sum of radii.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with hex initialization
    circles = _hex_init(N)
    best_circles = [c[:] for c in circles]
    best_score = _objective(best_circles)
    
    # Multiple restarts with different strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Choose initialization strategy
        if attempts % 3 == 1:
            init_circles = _hex_init(N)
        elif attempts % 3 == 2:
            init_circles = _random_init(N, seed=attempts)
        else:
            # Perturb the best found so far
            init_circles = [list(c) for c in best_circles]
            for i in range(N):
                init_circles[i][0] += random.gauss(0, 0.02)
                init_circles[i][1] += random.gauss(0, 0.02)
                init_circles[i][2] = max(0.01, init_circles[i][2] + random.gauss(0, 0.01))
                # Clamp to bounds
                x, y, r = init_circles[i]
                init_circles[i][0] = max(r, min(1 - r, x))
                init_circles[i][1] = max(r, min(1 - r, y))
        
        # Make sure initial is valid; if not, shrink radii
        if not _is_valid(init_circles):
            for i in range(N):
                init_circles[i][2] *= 0.8
            if not _is_valid(init_circles):
                init_circles = _hex_init(N)
        
        # Run local search
        cur_circles, cur_score = _local_search(init_circles, max_iters=3000, deadline=deadline)
        
        # Try to increase radii
        cur_circles, cur_score = _hill_climb_radii(cur_circles, max_iters=1500, deadline=deadline)
        
        # Fix any remaining overlaps
        cur_circles, cur_score = _pairwise_separation(cur_circles, max_iters=2000, deadline=deadline)
        
        # Another round of radius climbing
        cur_circles, cur_score = _hill_climb_radii(cur_circles, max_iters=1000, deadline=deadline)
        
        # Update best
        if cur_score > best_score:
            best_score = cur_score
            best_circles = [c[:] for c in cur_circles]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final validation
    if not _is_valid(best_circles):
        # Fall back to grid
        best_circles = _grid_init(N)
    
    # Convert to numpy array
    result = np.array(best_circles, dtype=np.float64)
    return result
# EVOLVE-BLOCK-END
