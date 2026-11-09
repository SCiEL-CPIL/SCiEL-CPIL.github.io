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


def _ls_step(circles, max_iter=5000, deadline=None):
    """Lubachevsky-Steampole step: expand the most constrained circle."""
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Find the circle with the smallest clearance
        min_clearance = np.inf
        min_idx = -1
        
        for i in range(n):
            clearance = np.inf
            
            # Check boundaries
            clearance = min(clearance, circles[i, 0] - circles[i, 2])
            clearance = min(clearance, 1 - circles[i, 0] - circles[i, 2])
            clearance = min(clearance, circles[i, 1] - circles[i, 2])
            clearance = min(clearance, 1 - circles[i, 1] - circles[i, 2])
            
            # Check other circles
            for j in range(n):
                if i == j:
                    continue
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                clearance = min(clearance, dist - circles[i, 2] - circles[j, 2])
            
            if clearance < min_clearance:
                min_clearance = clearance
                min_idx = i
        
        if min_idx == -1:
            break
            
        # Expand the most constrained circle
        if min_clearance > 0:
            circles[min_idx, 2] += min_clearance
        
        # If clearance is 0, jiggle the circle slightly
        else:
            # Random jiggle
            angle = np.random.uniform(0, 2 * np.pi)
            step = 0.01 * circles[min_idx, 2]
            circles[min_idx, 0] += step * np.cos(angle)
            circles[min_idx, 1] += step * np.sin(angle)
            
            # Keep inside bounds
            circles[min_idx, 0] = np.clip(circles[min_idx, 0], circles[min_idx, 2], 1 - circles[min_idx, 2])
            circles[min_idx, 1] = np.clip(circles[min_idx, 1], circles[min_idx, 2], 1 - circles[min_idx, 2])
            
            # Resolve overlaps by pushing apart
            for j in range(n):
                if min_idx == j:
                    continue
                dx = circles[min_idx, 0] - circles[j, 0]
                dy = circles[min_idx, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = circles[min_idx, 2] + circles[j, 2] - dist
                if overlap > 0 and dist > 1e-10:
                    ux = dx / dist
                    uy = dy / dist
                    circles[min_idx, 0] += 0.5 * overlap * ux
                    circles[min_idx, 1] += 0.5 * overlap * uy
                    circles[j, 0] -= 0.5 * overlap * ux
                    circles[j, 1] -= 0.5 * overlap * uy
    
    return circles


def _maximize_radii(circles, max_iter=1000, deadline=None):
    """Steepest ascent on radii with hard constraints."""
    circles = circles.copy()
    n = len(circles)
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try increasing each radius slightly
        improved = False
        for i in range(n):
            for dr in [0.005, 0.002, 0.001, 0.0005]:
                test = circles.copy()
                test[i, 2] += dr
                
                # Check validity
                valid = True
                x, y, r = test[:, 0], test[:, 1], test[:, 2]
                
                # Bounds
                if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                    valid = False
                
                # Overlaps
                if valid:
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                
                if valid:
                    circles = test
                    improved = True
                    break
        
        if not improved:
            break
        
        score = np.sum(circles[:, 2])
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
        
        # LS packing
        ls_result = _ls_step(current, max_iter=2000, deadline=deadline - 20)
        
        # Maximize radii
        final_result = _maximize_radii(ls_result, max_iter=500, deadline=deadline - 10)
        
        score = _objective(final_result)
        
        if score > best_score:
            best_score = score
            best_circles = final_result
    
    return best_circles
# EVOLVE-BLOCK-END
