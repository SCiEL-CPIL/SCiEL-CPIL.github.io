"""Circle packing in unit square using iterative maximal expansion (hard constraints)."""
import numpy as np
import time
import random

N = 26

def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
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

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _init_grid(n):
    """Initialize with a grid of small circles."""
    side = int(np.ceil(np.sqrt(n)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            circles.append([x, y, r])
    return np.array(circles[:n])

def _init_random(n):
    """Initialize with random positions and small radii."""
    circles = []
    for _ in range(n):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.01, 0.05)
        circles.append([x, y, r])
    return np.array(circles)

def _max_radius_at(x, y, circles, exclude_idx):
    """Find the maximum radius a circle at (x,y) can have without violating constraints."""
    max_r = min(x, 1 - x, y, 1 - y)
    n = len(circles)
    for i in range(n):
        if i == exclude_idx:
            continue
        dx = x - circles[i, 0]
        dy = y - circles[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - circles[i, 2])
    return max(0, max_r)

def _clearance(circles):
    """Compute clearance for each circle (how much it can grow)."""
    n = len(circles)
    clearances = np.zeros(n)
    for i in range(n):
        x, y, r = circles[i]
        max_r = _max_radius_at(x, y, circles, i)
        clearances[i] = max_r - r
    return clearances

def _greedy_expand(circles, deadline, temp=0.0):
    """
    Lubachevsky-Steampole style expansion with optional simulated annealing.
    """
    n = len(circles)
    best_circles = circles.copy()
    best_score = _sum_radii(best_circles)
    
    # Precompute initial clearances
    clearances = _clearance(circles)
    
    # Main loop
    while time.time() < deadline:
        # Find circle with max clearance
        max_idx = np.argmax(clearances)
        max_clearance = clearances[max_idx]
        
        if max_clearance < 1e-6:
            # All circles are packed, try a random kick
            if temp > 0:
                # Simulated annealing: randomly kick a circle
                kick_idx = np.random.randint(n)
                dx = random.gauss(0, 0.01 * temp)
                dy = random.gauss(0, 0.01 * temp)
                circles[kick_idx, 0] += dx
                circles[kick_idx, 1] += dy
                # Clamp to bounds
                circles[kick_idx, 0] = np.clip(circles[kick_idx, 0], 0, 1)
                circles[kick_idx, 1] = np.clip(circles[kick_idx, 1], 0, 1)
                # Recompute clearances
                clearances = _clearance(circles)
                continue
            else:
                break
        
        # Expand the circle
        x, y, r = circles[max_idx]
        new_r = r + max_clearance
        circles[max_idx, 2] = new_r
        
        # Resolve collisions: push neighbors away
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if i == max_idx:
                    continue
                dx = circles[i, 0] - x
                dy = circles[i, 1] - y
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = circles[i, 2] + new_r - dist
                if overlap > 1e-9:
                    if dist < 1e-9:
                        # Random direction
                        angle = random.uniform(0, 2 * np.pi)
                        dx, dy = np.cos(angle), np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    # Push neighbor away
                    circles[i, 0] += dx * overlap
                    circles[i, 1] += dy * overlap
                    changed = True
            
            # Clamp to bounds and resolve boundary collisions
            for i in range(n):
                if circles[i, 0] < circles[i, 2]:
                    circles[i, 0] = circles[i, 2]
                    changed = True
                if circles[i, 0] > 1 - circles[i, 2]:
                    circles[i, 0] = 1 - circles[i, 2]
                    changed = True
                if circles[i, 1] < circles[i, 2]:
                    circles[i, 1] = circles[i, 2]
                    changed = True
                if circles[i, 1] > 1 - circles[i, 2]:
                    circles[i, 1] = 1 - circles[i, 2]
                    changed = True
        
        # Recompute clearances
        clearances = _clearance(circles)
        
        # Update best
        score = _sum_radii(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    return best_circles

def _local_jiggle(circles, deadline, temp=0.01):
    """Local optimization with small random perturbations."""
    n = len(circles)
    best_circles = circles.copy()
    best_score = _sum_radii(best_circles)
    
    while time.time() < deadline:
        # Randomly pick a circle and jiggle it
        idx = np.random.randint(n)
        dx = random.gauss(0, temp)
        dy = random.gauss(0, temp)
        circles[idx, 0] += dx
        circles[idx, 1] += dy
        
        # Clamp to bounds
        circles[idx, 0] = np.clip(circles[idx, 0], 0, 1)
        circles[idx, 1] = np.clip(circles[idx, 1], 0, 1)
        
        # Resolve collisions
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = circles[i, 2] + circles[j, 2] - dist
                    if overlap > 1e-9:
                        if dist < 1e-9:
                            angle = random.uniform(0, 2 * np.pi)
                            dx, dy = np.cos(angle), np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        # Push apart
                        circles[i, 0] += dx * overlap / 2
                        circles[i, 1] += dy * overlap / 2
                        circles[j, 0] -= dx * overlap / 2
                        circles[j, 1] -= dy * overlap / 2
                        changed = True
            
            # Clamp to bounds
            for i in range(n):
                if circles[i, 0] < circles[i, 2]:
                    circles[i, 0] = circles[i, 2]
                    changed = True
                if circles[i, 0] > 1 - circles[i, 2]:
                    circles[i, 0] = 1 - circles[i, 2]
                    changed = True
                if circles[i, 1] < circles[i, 2]:
                    circles[i, 1] = circles[i, 2]
                    changed = True
                if circles[i, 1] > 1 - circles[i, 2]:
                    circles[i, 1] = 1 - circles[i, 2]
                    changed = True
        
        # Try to expand all circles
        for i in range(n):
            max_r = _max_radius_at(circles[i, 0], circles[i, 1], circles, i)
            if max_r > circles[i, 2]:
                circles[i, 2] = max_r
        
        score = _sum_radii(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    return best_circles

def construct():
    """Construct optimal circle packing in unit square."""
    deadline = time.time() + 140  # Safety margin
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = [
        _init_grid(N),
        _init_random(N),
        _init_random(N),
        _init_random(N),
    ]
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        # Greedy expansion
        expanded = _greedy_expand(init.copy(), deadline - 20, temp=0.01)
        
        # Local jiggle
        if time.time() < deadline - 10:
            optimized = _local_jiggle(expanded.copy(), deadline - 5, temp=0.005)
        else:
            optimized = expanded
        
        score = _sum_radii(optimized)
        if score > best_score:
            best_score = score
            best_circles = optimized
    
    if best_circles is None:
        best_circles = _init_grid(N)
    
    return best_circles
