# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _check_validity(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    # Check finite
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    # Check side > 0
    if np.any(sides <= 0):
        return False
    # Check inside unit square (with tol)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check no overlap
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _penalized_objective(centers, sides, penalty_weight=500.0):
    """
    Smooth penalized objective for optimization.
    Returns sum(sides) - penalty for violations.
    We want to maximize this.
    """
    n = len(sides)
    obj = np.sum(sides)
    
    # Boundary penalty: squares must be inside [0, 1] x [0, 1]
    # We penalize any part of the square that goes outside.
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Distance outside boundary
        d_left = max(0, s/2 - cx)
        d_right = max(0, cx + s/2 - 1)
        d_bottom = max(0, s/2 - cy)
        d_top = max(0, cy + s/2 - 1)
        # Quadratic penalty for boundary violations
        obj -= penalty_weight * (d_left**2 + d_right**2 + d_bottom**2 + d_top**2)
    
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            t = (sides[i] + sides[j])/2
            # Overlap in x: max(0, t - dx)
            # Overlap in y: max(0, t - dy)
            ox = max(0, t - dx)
            oy = max(0, t - dy)
            # We want to penalize when both are positive (actual overlap)
            # Using product ox*oy approximates the area of overlap if aligned, 
            # but here we just need a strong penalty when they overlap.
            # A better penalty is max(0, t - max(dx, dy)) which is the minimum distance to separate them.
            # Let's use the sum of overlaps squared for smoothness.
            overlap_penalty = ox**2 + oy**2
            obj -= penalty_weight * overlap_penalty
            
    return obj

def _simulate_annealing(centers, sides, time_limit=10.0, initial_temp=0.05, cooling_rate=0.999):
    """
    Run Simulated Annealing to optimize the packing.
    """
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _penalized_objective(best_centers, best_sides)
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_obj = _penalized_objective(current_centers, current_sides)
    
    temp = initial_temp
    start_time = time.time()
    n = len(sides)
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        # Move a random square
        i = np.random.randint(0, n)
        
        # Perturb center
        delta_cx = np.random.normal(0, 0.01)
        delta_cy = np.random.normal(0, 0.01)
        new_centers[i, 0] += delta_cx
        new_centers[i, 1] += delta_cy
        
        # Perturb side length
        delta_s = np.random.normal(0, 0.01)
        new_sides[i] += delta_s
        
        # Keep side positive and not too large
        new_sides[i] = np.clip(new_sides[i], 0.01, 0.6)
        
        # Keep centers within a reasonable range (e.g., [-0.5, 1.5])
        new_centers[i, 0] = np.clip(new_centers[i, 0], -0.5, 1.5)
        new_centers[i, 1] = np.clip(new_centers[i, 1], -0.5, 1.5)
        
        new_obj = _penalized_objective(new_centers, new_sides)
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        if delta_obj > 0:
            # Always accept improvement
            current_centers = new_centers
            current_sides = new_sides
            current_obj = new_obj
        else:
            # Accept worse solution with probability exp(delta_obj / temp)
            if np.random.rand() < np.exp(delta_obj / temp):
                current_centers = new_centers
                current_sides = new_sides
                current_obj = new_obj
        
        # Update best
        if current_obj > best_obj:
            best_centers = current_centers.copy()
            best_sides = current_sides.copy()
            best_obj = current_obj
        
        # Cool down
        temp *= cooling_rate
        if temp < 1e-6:
            break
            
    return best_centers, best_sides

def _grid_packing():
    """Baseline grid packing."""
    n = 10
    k = 4  # 4x4 grid, use 10 cells
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

def _random_packing():
    """Random initial packing."""
    centers = np.random.uniform(0, 1, size=(10, 2))
    sides = np.random.uniform(0.05, 0.2, size=10)
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Try different initializations
    initializations = [
        _grid_packing(),
        _random_packing(),
        _random_packing()
    ]
    
    # Add a "corner" initialization: place large squares in corners
    centers_corner = np.zeros((10, 2))
    sides_corner = np.zeros(10)
    corners = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for i in range(4):
        centers_corner[i] = corners[i]
        sides_corner[i] = 0.4
    for i in range(4, 10):
        centers_corner[i] = np.random.uniform(0.2, 0.8, size=2)
        sides_corner[i] = 0.15
    initializations.append((centers_corner, sides_corner))
    
    for init_centers, init_sides in initializations:
        if time.time() > deadline:
            break
            
        # Run Simulated Annealing
        c_opt, s_opt = _simulate_annealing(init_centers, init_sides, time_limit=60.0)
        
        # Check validity and update best
        if _check_validity(c_opt, s_opt):
            score = np.sum(s_opt)
            if score > best_score:
                best_score = score
                best_centers = c_opt.copy()
                best_sides = s_opt.copy()
                
    # If no valid solution found, fallback to grid
    if best_centers is None:
        best_centers, best_sides = _grid_packing()
        
    # Final check
    if not _check_validity(best_centers, best_sides):
        best_centers, best_sides = _grid_packing()
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
