"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-END

def _validate_and_fix(xyz):
    """Ensure circles are inside [0,1]^2 with positive radii. Returns (xyz, valid)."""
    xyz = np.array(xyz, dtype=float).copy()
    if xyz.shape != (N, 3):
        return None, False
    # Clip positions to [0,1]
    xyz[:, 0] = np.clip(xyz[:, 0], 0.0, 1.0)
    xyz[:, 1] = np.clip(xyz[:, 1], 0.0, 1.0)
    # Ensure radii positive
    xyz[:, 2] = np.clip(xyz[:, 2], 1e-6, 0.5)
    return xyz, True

def _check_validity(xyz, tol=1e-7):
    """Check if all circles are disjoint and inside the unit square."""
    if xyz is None:
        return False
    for i in range(N):
        x, y, r = xyz[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r <= 0:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = xyz[i, 0] - xyz[j, 0]
            dy = xyz[i, 1] - xyz[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < xyz[i, 2] + xyz[j, 2] - tol:
                return False
    return True

def _score(xyz):
    """Compute the objective: sum of radii."""
    if xyz is None:
        return 0.0
    return float(np.sum(xyz[:, 2]))

def _max_radius_at_point(x, y, existing_xyz, tol=1e-9):
    """Find the maximum radius for a circle centered at (x,y) given existing circles."""
    # Distance to boundaries
    r_boundary = min(x, 1.0 - x, y, 1.0 - y)
    
    # Distance to existing circles
    r_existing = r_boundary
    if len(existing_xyz) > 0:
        for ex, ey, er in existing_xyz:
            dist = np.sqrt((x - ex)**2 + (y - ey)**2)
            r_existing = min(r_existing, dist - er)
    
    return max(0.0, r_existing - tol)

def _find_best_position_for_radius(target_r, existing_xyz, n_samples=200, deadline=None):
    """
    Find a position where a circle of radius target_r fits, maximizing clearance.
    Uses a grid search + local refinement.
    """
    best_pos = None
    best_clearance = -1
    
    # Grid search
    grid_n = 15
    xs = np.linspace(target_r, 1.0 - target_r, grid_n)
    ys = np.linspace(target_r, 1.0 - target_r, grid_n)
    
    for x in xs:
        for y in ys:
            # Check if circle of radius target_r fits
            if x < target_r or x > 1 - target_r or y < target_r or y > 1 - target_r:
                continue
            
            # Check against existing circles
            valid = True
            for ex, ey, er in existing_xyz:
                dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                if dist < target_r + er - 1e-9:
                    valid = False
                    break
            
            if valid:
                # Compute clearance (min distance to boundary and other circles minus radii)
                clearance = min(x, 1-x, y, 1-y) - target_r
                for ex, ey, er in existing_xyz:
                    dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                    clearance = min(clearance, dist - target_r - er)
                
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_pos = (x, y)
    
    # Local refinement around best position
    if best_pos is not None:
        x0, y0 = best_pos
        # Try small perturbations
        for dx in np.linspace(-0.02, 0.02, 5):
            for dy in np.linspace(-0.02, 0.02, 5):
                x, y = x0 + dx, y0 + dy
                if x < target_r or x > 1 - target_r or y < target_r or y > 1 - target_r:
                    continue
                
                valid = True
                for ex, ey, er in existing_xyz:
                    dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                    if dist < target_r + er - 1e-9:
                        valid = False
                        break
                
                if valid:
                    clearance = min(x, 1-x, y, 1-y) - target_r
                    for ex, ey, er in existing_xyz:
                        dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                        clearance = min(clearance, dist - target_r - er)
                    
                    if clearance > best_clearance:
                        best_clearance = clearance
                        best_pos = (x, y)
    
    return best_pos, best_clearance

def _greedy_insertion(deadline, seed=42):
    """
    Sequential greedy insertion: place circles one by one, each as large as possible.
    """
    rng = np.random.RandomState(seed)
    xyz = []
    
    # Start with a larger circle in the center, then add smaller ones
    # First circle: place at center with max radius
    x0, y0 = 0.5, 0.5
    r0 = 0.5
    xyz.append([x0, y0, r0])
    
    # Add remaining circles greedily
    for i in range(1, N):
        if time.time() > deadline - 5.0:
            break
        
        # Try to place next circle
        # Start with a reasonable radius estimate
        # The remaining space is roughly (1 - 2*0.5)^2 / (N-1) for area, but that's too small
        # Let's try radii from large to small
        
        # Estimate max possible radius for remaining circles
        # Area argument: total area = sum(pi*r^2) <= 1
        # If first circle has r=0.5, area used = pi*0.25 ≈ 0.785
        # Remaining area ≈ 0.215 for 25 circles, so avg r ≈ sqrt(0.215/(25*pi)) ≈ 0.052
        
        # Try a range of radii
        radii_to_try = []
        # Start with larger radii and decrease
        for r_try in [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]:
            pos, clearance = _find_best_position_for_radius(r_try, xyz)
            if pos is not None:
                radii_to_try.append((r_try, pos, clearance))
        
        if not radii_to_try:
            # Couldn't place any circle, break
            break
        
        # Choose the one with largest radius (or best clearance if tied)
        # Actually, let's choose the largest radius that fits
        best_r, best_pos, best_clear = max(radii_to_try, key=lambda x: x[0])
        
        xyz.append([best_pos[0], best_pos[1], best_r])
    
    # Convert to array and pad if needed
    while len(xyz) < N:
        # Add very small circles if we didn't reach N
        xyz.append([0.5, 0.5, 0.01])
    
    return np.array(xyz[:N])

def _local_relaxation(xyz_init, deadline, max_iters=100):
    """
    Local relaxation: for each circle, try to increase its radius by moving it slightly.
    Uses scipy's minimize to optimize each circle's position and radius.
    """
    xyz = xyz_init.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            
            x_i, y_i, r_i = xyz[i]
            
            # Objective: maximize r_i subject to constraints
            # Constraints: 
            # 1. Circle i stays in [0,1]^2: x_i - r_i >= 0, x_i + r_i <= 1, etc.
            # 2. Circle i doesn't overlap with any other circle j
            
            def objective(params):
                x, y, r = params
                # Penalty for being out of bounds
                penalty = 0
                if x - r < 0:
                    penalty += (x - r) ** 2
                if x + r > 1:
                    penalty += (x + r - 1) ** 2
                if y - r < 0:
                    penalty += (y - r) ** 2
                if y + r > 1:
                    penalty += (y + r - 1) ** 2
                
                # Penalty for overlapping with other circles
                for j in range(N):
                    if j == i:
                        continue
                    xj, yj, rj = xyz[j]
                    dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                    if dist < r + rj:
                        penalty += (r + rj - dist) ** 2
                
                # We want to maximize r, so minimize -r + penalty
                return -r + penalty
            
            # Bounds
            bounds = [(0, 1), (0, 1), (0.01, 0.5)]
            
            # Initial guess
            x0 = [x_i, y_i, r_i]
            
            # Try to optimize
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 50, 'ftol': 1e-8})
                
                if res.success and res.fun < objective(x0) - 1e-6:
                    new_x, new_y, new_r = res.x
                    # Check if this is valid
                    xyz_trial = xyz.copy()
                    xyz_trial[i] = [new_x, new_y, new_r]
                    
                    if _check_validity(xyz_trial):
                        xyz = xyz_trial
                        improved = True
            except:
                pass
        
        if not improved:
            break
    
    return xyz

def _construct_with_greedy_and_relax(deadline):
    """
    Main construction: greedy insertion followed by local relaxation.
    """
    # Try multiple seeds for greedy insertion
    best_xyz = None
    best_score = 0.0
    
    seeds = [42, 123, 999, 7, 314]
    time_per_seed = (deadline - time.time() - 10.0) / len(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break
        
        # Greedy insertion
        xyz_greedy = _greedy_insertion(deadline, seed=seed)
        
        # Local relaxation
        xyz_relaxed = _local_relaxation(xyz_greedy, deadline, max_iters=50)
        
        # Validate
        if _check_validity(xyz_relaxed):
            score = _score(xyz_relaxed)
            if score > best_score:
                best_score = score
                best_xyz = xyz_relaxed
    
    return best_xyz, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s kill
    
    best_xyz, best_score = _construct_with_greedy_and_relax(deadline)
    
    if best_xyz is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
        best_xyz = np.array(cells[:N])
    
    # Final validation
    best_xyz, valid = _validate_and_fix(best_xyz)
    if not valid or not _check_validity(best_xyz):
        # Last resort: very small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                cells.append([x, y, r])
        best_xyz = np.array(cells[:N])
    
    return best_xyz

# EVOLVE-BLOCK-END
