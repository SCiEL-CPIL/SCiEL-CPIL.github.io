"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_valid(circles, tol=1e-9):
    """Check if all circles are inside [0,1]^2 with disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _objective(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])

def _solve_one_restart(seed, time_budget=10.0):
    """Solve one instance with a given random seed."""
    rng = np.random.RandomState(seed)
    
    # Initialize with a grid pattern with some randomness
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            y = (j + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            r = 0.5 / side * rng.uniform(0.8, 1.0)
            cells.append([x, y, r])
    circles = np.array(cells[:N], dtype=np.float64)
    
    # Ensure initial validity by shrinking if needed
    for _ in range(100):
        valid = True
        for i in range(N):
            x, y, r = circles[i]
            # Check boundaries
            min_dist = min(x, 1-x, y, 1-y)
            if r > min_dist:
                r = min_dist * 0.999
                circles[i][2] = r
                valid = False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                req = circles[i][2] + circles[j][2]
                if dist < req:
                    # Shrink the smaller circle
                    if circles[i][2] <= circles[j][2]:
                        circles[i][2] = max(0.001, (dist - circles[j][2]) * 0.999)
                    else:
                        circles[j][2] = max(0.001, (dist - circles[i][2]) * 0.999)
                    valid = False
        if valid:
            break
    
    best_circles = circles.copy()
    best_obj = _objective(circles)
    
    start_time = time.time()
    iter_count = 0
    step_size = 0.05
    
    while time.time() - start_time < time_budget:
        iter_count += 1
        
        # Randomly pick a circle to perturb
        idx = rng.randint(N)
        old_c = circles[idx].copy()
        
        # Try moving in a random direction
        angle = rng.uniform(0, 2*np.pi)
        dist = step_size * rng.uniform(0.5, 1.5)
        circles[idx][0] += dist * np.cos(angle)
        circles[idx][1] += dist * np.sin(angle)
        
        # Try increasing radius slightly
        if rng.random() < 0.3:
            circles[idx][2] *= 1.01
        
        # Check validity and repair
        new_obj = _objective(circles)
        
        if _is_valid(circles):
            if new_obj > best_obj:
                best_obj = new_obj
                best_circles = circles.copy()
            step_size = max(0.001, step_size * 0.99)
        else:
            # Revert
            circles[idx] = old_c
            step_size = max(0.001, step_size * 0.95)
        
        # Occasionally try to increase all radii slightly
        if iter_count % 50 == 0:
            old_circles = circles.copy()
            for i in range(N):
                circles[i][2] *= 1.005
            if _is_valid(circles):
                new_obj = _objective(circles)
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_circles = circles.copy()
            else:
                circles = old_circles
    
    return best_circles, best_obj

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_circles = None
    best_obj = -np.inf
    
    # Try multiple restarts
    for seed in range(100):
        if time.time() - start_time > deadline:
            break
        
        time_per_restart = max(1.0, (deadline - (time.time() - start_time)) / max(1, 100 - seed))
        circles, obj = _solve_one_restart(seed, time_budget=time_per_restart)
        
        if obj > best_obj:
            best_obj = obj
            best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
