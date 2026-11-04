"""
Strategy: Physics-Based Discrete Event Simulation (Molecular Dynamics)
Instead of optimizing (x, y, r) directly, we treat circles as particles in a box.
We start with a valid packing and then apply a "pressure" force to the walls 
and between circles. The system evolves to minimize potential energy.
We define a custom potential where the "rest length" is 0 (hard core) but we 
add a bias that pushes circles to be as large as possible by effectively 
increasing their size until they hit constraints.

Actually, a simpler and more robust physics approach for "max sum of radii":
1. Start with a valid packing of 26 circles (e.g., hexagonal or grid).
2. Iteratively:
   a. For each circle, calculate the maximum radius it can have given current positions of others.
   b. Set radius to this max.
   c. Apply forces: 
      - Repulsive force between overlapping circles (if any, though we set to max so they touch).
      - Attractive force to "fill" space? No, that's hard.
      
Let's refine the physics approach:
The problem is equivalent to finding the configuration that maximizes sum(r_i).
This is a non-convex optimization.

Alternative Structural Change: **Simulated Annealing on the configuration space with adaptive radius assignment.**
Current: Gradient Descent + Grid Search.
New: Simulated Annealing where the "energy" is the negative sum of radii, and moves are random perturbations of positions AND radii.
Key difference: Global search via SA vs Local search via GD.

Let's implement a robust Simulated Annealing approach:
1. State: positions (2N) and radii (N).
2. Move: Perturb one or more circles' positions and radii.
3. Energy: -sum(radii) + large_penalty(violations).
4. Annealing schedule: Exponential decay.
5. Initial state: Hexagonal packing scaled to fit.

This is materially different from L-BFGS-B + Grid Search because:
- It is a stochastic global search method.
- It can escape local minima that trap gradient-based methods.
- It handles the non-convexity of the feasible region more naturally by accepting worse states temporarily.
"""

import numpy as np
import time

N = 26

def _hexagonal_init(n):
    """
    Generate a hexagonal packing of n circles inside [0,1]^2.
    Returns (xy, r) where xy is (n,2) and r is (n,).
    """
    # Approximate number of rows
    # Hex packing density is ~0.9069
    # Area per circle ~ pi/2 * r^2 * (2/sqrt(3))
    # Let's just place them in a hex grid and scale.
    
    # Generate grid points
    points = []
    # Estimate rows based on n
    # For hex, roughly n ~ 1.5 * rows^2
    rows = int(np.ceil(np.sqrt(n / 1.5))) + 1
    
    for row in range(rows):
        for col in range(rows + 1):
            if len(points) >= n:
                break
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            y = row * 0.866  # sqrt(3)/2
            points.append((x, y))
            if len(points) >= n:
                break
    
    points = np.array(points[:n])
    
    # Normalize to fit in [0,1]^2
    # Find min and max
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # Scale factor
    scale_x = 1.0 / (max_x - min_x) if max_x > min_x else 1.0
    scale_y = 1.0 / (max_y - min_y) if max_y > min_y else 1.0
    scale = min(scale_x, scale_y)
    
    # Center it
    points[:, 0] = (points[:, 0] - min_x) * scale + 0.5 - (max_x - min_x) * scale / 2
    points[:, 1] = (points[:, 1] - min_y) * scale + 0.5 - (max_y - min_y) * scale / 2
    
    # Calculate max radius for each circle based on neighbors and walls
    # Initial radius: small
    r = np.full(n, 0.01)
    
    # Refine radii to be as large as possible
    for _ in range(20):
        for i in range(n):
            max_r = min(points[i, 0], 1.0 - points[i, 0], points[i, 1], 1.0 - points[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    max_r = min(max_r, (dist - r[j]) / 2.0)
            r[i] = max(0.001, max_r)
            
    return points, r

def _random_init(n):
    """Generate a random valid packing."""
    # Simple: place circles one by one at random valid positions
    xy = np.zeros((n, 2))
    r = np.zeros(n)
    
    # Start with a small radius and try to place
    # This is slow and might fail. Let's use a grid and jitter.
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        gx = i % side
        gy = i // side
        x = (gx + 0.5) / side + np.random.uniform(-0.1/side, 0.1/side)
        y = (gy + 0.5) / side + np.random.uniform(-0.1/side, 0.1/side)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        xy[i] = [x, y]
        r[i] = 0.02
        
    # Expand radii
    for _ in range(50):
        for i in range(n):
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(xy[i] - xy[j])
                    max_r = min(max_r, (dist - r[j]) / 2.0)
            r[i] = max(0.001, max_r)
            
    return xy, r

def _score(xy, r):
    """
    Calculate the score (sum of radii) if valid, else -infinity.
    Also returns penalty for invalidity to guide optimization.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        if xy[i, 0] < r[i] or 1.0 - xy[i, 0] < r[i] or xy[i, 1] < r[i] or 1.0 - xy[i, 1] < r[i]:
            return -np.inf, 0.0
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(xy[i] - xy[j])
            if dist < r[i] + r[j]:
                return -np.inf, 0.0
                
    return np.sum(r), np.sum(r)

def _penalized_score(xy, r, penalty_weight=1000.0):
    """
    Score with penalties for violations. Used for annealing acceptance.
    """
    n = len(xy)
    penalty = 0.0
    sum_r = 0.0
    
    for i in range(n):
        sum_r += r[i]
        # Wall penalties
        if xy[i, 0] < r[i]:
            penalty += (r[i] - xy[i, 0])**2
        if 1.0 - xy[i, 0] < r[i]:
            penalty += (r[i] - (1.0 - xy[i, 0]))**2
        if xy[i, 1] < r[i]:
            penalty += (r[i] - xy[i, 1])**2
        if 1.0 - xy[i, 1] < r[i]:
            penalty += (r[i] - (1.0 - xy[i, 1]))**2
            
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(xy[i] - xy[j])
            if dist < r[i] + r[j]:
                penalty += (r[i] + r[j] - dist)**2
                
    return sum_r - penalty_weight * penalty

def _anneal(xy, r, start_temp, end_temp, steps, deadline):
    """
    Simulated Annealing.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    current_score, _ = _score(current_xy, current_r)
    if np.isinf(current_score):
        # Fix initial state
        for i in range(n):
            current_r[i] = 0.01
            current_xy[i, 0] = np.clip(current_xy[i, 0], current_r[i], 1.0-current_r[i])
            current_xy[i, 1] = np.clip(current_xy[i, 1], current_r[i], 1.0-current_r[i])
        current_score, _ = _score(current_xy, current_r)
        
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_score = current_score
    if np.isinf(best_score):
        best_score = 0.0
    else:
        best_score = current_score
        
    # We want to MAXIMIZE sum(r), so minimize -sum(r).
    # Annealing usually minimizes energy. Let Energy = -sum(r) + Penalty.
    
    def energy(xy, r):
        s, _ = _score(xy, r)
        if np.isinf(s):
            # Calculate penalty
            p = 0.0
            for i in range(n):
                if xy[i, 0] < r[i]: p += (r[i] - xy[i, 0])**2
                if 1.0 - xy[i, 0] < r[i]: p += (r[i] - (1.0 - xy[i, 0]))**2
                if xy[i, 1] < r[i]: p += (r[i] - xy[i, 1])**2
                if 1.0 - xy[i, 1] < r[i]: p += (r[i] - (1.0 - xy[i, 1]))**2
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(xy[i] - xy[j])
                    if dist < r[i] + r[j]:
                        p += (r[i] + r[j] - dist)**2
            return -s + 1000.0 * p # s is -inf, so -s is +inf, but we want finite penalty
            # Actually, if invalid, return large positive energy
            return 1000.0 * p + 100.0
        return -s # Minimize negative sum of radii
        
    current_energy = energy(current_xy, current_r)
    best_energy = current_energy
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Generate move
        # 1. Perturb position of one circle
        i = np.random.randint(n)
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        
        new_xy[i, 0] += dx
        new_xy[i, 1] += dy
        
        # 2. Optionally perturb radius
        if np.random.random() < 0.5:
            dr = np.random.normal(0, 0.005)
            new_r[i] = max(0.001, new_r[i] + dr)
            
        # Clip position to box roughly
        new_xy[i, 0] = np.clip(new_xy[i, 0], 0.0, 1.0)
        new_xy[i, 1] = np.clip(new_xy[i, 1], 0.0, 1.0)
        
        new_energy = energy(new_xy, new_r)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
            current_xy = new_xy
            current_r = new_r
            current_energy = new_energy
            
            # Check if best
            s, _ = _score(current_xy, current_r)
            if not np.isinf(s) and s > best_score:
                best_score = s
                best_xy = current_xy.copy()
                best_r = current_r.copy()
                
    return best_xy, best_r, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_xy = None
    best_r = None
    best_score = 0.0
    
    # Multiple initializations
    inits = []
    
    # 1. Hexagonal
    try:
        xy, r = _hexagonal_init(N)
        inits.append((xy, r))
    except:
        pass
        
    # 2. Random
    try:
        xy, r = _random_init(N)
        inits.append((xy, r))
    except:
        pass
        
    # 3. Grid
    side = int(np.ceil(np.sqrt(N)))
    xy_grid = np.zeros((N, 2))
    r_grid = np.zeros(N)
    for i in range(N):
        gx = i % side
        gy = i // side
        xy_grid[i, 0] = (gx + 0.5) / side
        xy_grid[i, 1] = (gy + 0.5) / side
        r_grid[i] = 0.4 / side
    inits.append((xy_grid, r_grid))
    
    # Run annealing on each
    for idx, (xy, r) in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Check initial validity
        s, _ = _score(xy, r)
        if np.isinf(s):
            # Try to fix
            for i in range(N):
                max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                for j in range(N):
                    if i != j:
                        dist = np.linalg.norm(xy[i] - xy[j])
                        max_r = min(max_r, (dist - r[j]) / 2.0)
                r[i] = max(0.001, max_r)
            s, _ = _score(xy, r)
            
        if not np.isinf(s) and s > best_score:
            best_score = s
            best_xy = xy.copy()
            best_r = r.copy()
            
        # Run annealing
        # Allocate time based on remaining
        remaining = deadline - time.time()
        steps = int(remaining * 100) # 100 steps per second
        if steps < 100:
            steps = 100
            
        start_temp = 0.1
        end_temp = 0.001
        
        a_xy, a_r, a_score = _anneal(xy, r, start_temp, end_temp, steps, deadline - 2.0)
        
        if a_score > best_score:
            best_score = a_score
            best_xy = a_xy
            best_r = a_r
            
    # Final polish: Local search on the best result
    if best_xy is not None and best_r is not None and time.time() < deadline - 5.0:
        # Simple local search: try to increase radii by moving circles
        for _ in range(100): # 100 iterations
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                # Try to move circle i slightly to allow larger radius?
                # Or just check if we can increase radius
                max_r = min(best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
                for j in range(N):
                    if i != j:
                        dist = np.linalg.norm(best_xy[i] - best_xy[j])
                        max_r = min(max_r, (dist - best_r[j]) / 2.0)
                        
                if max_r > best_r[i] + 1e-6:
                    best_r[i] = max_r
                    improved = True
                    
            if not improved:
                break
                
        # Re-score
        s, _ = _score(best_xy, best_r)
        if not np.isinf(s) and s > best_score:
            best_score = s
            
    # Construct output
    result = np.zeros((N, 3))
    if best_xy is not None:
        for i in range(N):
            result[i, 0] = best_xy[i, 0]
            result[i, 1] = best_xy[i, 1]
            result[i, 2] = best_r[i]
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
