# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def _is_feasible(centers, sides, tol=1e-7):
    """Check if all squares are inside the unit square and non-overlapping."""
    n = len(sides)
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _sum_sides(sides):
    return np.sum(sides)

def _objective(x):
    """Objective to minimize: negative sum of sides + large penalty for infeasibility."""
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Ensure sides are positive
    if np.any(sides <= 0):
        return 1e10
    
    penalty = 0.0
    tol = 1e-7
    
    # Bounds penalty
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol:
            penalty += (-tol - (cx - s/2)) ** 2 * 100
        if cx + s/2 > 1 + tol:
            penalty += ((cx + s/2) - (1 + tol)) ** 2 * 100
        if cy - s/2 < -tol:
            penalty += (-tol - (cy - s/2)) ** 2 * 100
        if cy + s/2 > 1 + tol:
            penalty += ((cy + s/2) - (1 + tol)) ** 2 * 100
    
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                # Penalty for overlap
                penalty += (t - dx) ** 2 * 100 + (t - dy) ** 2 * 100
    
    return -_sum_sides(sides) + penalty

def _refine(centers, sides):
    """Refine a feasible packing using local optimization."""
    n = 10
    x0 = np.concatenate([centers.flatten(), sides])
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # cx
        bounds.append((0, 1))  # cy
    for i in range(n):
        bounds.append((0.01, 1.0))  # sides
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try Nelder-Mead
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.success or _objective(res.x) < best_val:
            if _is_feasible(res.x[:2*n].reshape(n, 2), res.x[2*n:]):
                best_x = res.x.copy()
                best_val = _objective(res.x)
    except:
        pass
    
    # Try Powell
    try:
        res = minimize(_objective, best_x, method='Powell',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if _is_feasible(res.x[:2*n].reshape(n, 2), res.x[2*n:]):
            if _objective(res.x) < best_val:
                best_x = res.x.copy()
                best_val = _objective(res.x)
    except:
        pass
    
    return best_x[:2*n].reshape(n, 2), best_x[2*n:]

def squares_in_square10() -> tuple:
    """
    Pack 10 squares in a unit square to maximize the sum of side lengths.
    Uses a multi-start local optimization approach.
    """
    start_time = time.time()
    deadline = start_time + 600  # 10 minute deadline
    
    n = 10
    best_centers = None
    best_sides = None
    best_sum = 0.0
    
    # Initial good heuristic packing for 10 squares
    # Known good packing: mix of different sized squares
    # Let's use several initial configurations
    
    initial_configs = []
    
    # Config 1: Grid-like but with some variation
    # 4x3 grid with some squares merged
    c1 = np.array([
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375],
        [0.125, 0.625], [0.375, 0.625]
    ])
    s1 = np.array([0.25]*10)
    initial_configs.append((c1, s1))
    
    # Config 2: Larger squares with smaller ones
    c2 = np.array([
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5], [0.125, 0.5], [0.875, 0.5], [0.5, 0.125], [0.5, 0.875],
        [0.5, 0.5]
    ])
    s2 = np.array([0.5, 0.5, 0.5, 0.5, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    initial_configs.append((c2, s2))
    
    # Config 3: Random feasible starting points
    rng = np.random.RandomState(42)
    for _ in range(20):
        # Generate random feasible configuration
        sides = rng.uniform(0.05, 0.4, n)
        # Sort by size descending
        order = np.argsort(-sides)
        sides_sorted = sides[order]
        
        centers = np.zeros((n, 2))
        for idx, i in enumerate(order):
            s = sides_sorted[idx]
            # Place in a grid-like manner
            row = idx // 4
            col = idx % 4
            cx = (col + 0.5) * (1.0/4) * 0.9 + 0.05
            cy = (row + 0.5) * (1.0/3) * 0.9 + 0.05
            # Adjust to fit
            cx = np.clip(cx, s/2, 1 - s/2)
            cy = np.clip(cy, s/2, 1 - s/2)
            centers[i] = [cx, cy]
        
        if _is_feasible(centers, sides):
            initial_configs.append((centers, sides))
    
    # Evaluate initial configs
    for centers, sides in initial_configs:
        if _is_feasible(centers, sides):
            s = _sum_sides(sides)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Local refinement
    for centers, sides in initial_configs[:15]:  # Limit number of refinements
        if time.time() > deadline - 60:
            break
        if not _is_feasible(centers, sides):
            continue
        try:
            new_centers, new_sides = _refine(centers, sides)
            if _is_feasible(new_centers, new_sides):
                s = _sum_sides(new_sides)
                if s > best_sum:
                    best_sum = s
                    best_centers = new_centers.copy()
                    best_sides = new_sides.copy()
        except:
            pass
    
    # Differential evolution for global search if time permits
    if time.time() < deadline - 120 and best_sum < 2.5:
        n = 10
        bounds = []
        for i in range(n):
            bounds.append((0.01, 0.99))  # cx
            bounds.append((0.01, 0.99))  # cy
        for i in range(n):
            bounds.append((0.01, 0.99))  # sides
        
        def de_objective(x):
            centers = x[:2*n].reshape(n, 2)
            sides = x[2*n:]
            if not _is_feasible(centers, sides):
                return 1e10
            return -_sum_sides(sides)
        
        try:
            result = differential_evolution(de_objective, bounds, maxiter=50, 
                                           popsize=15, seed=42, tol=1e-8)
            if result.success:
                centers = result.x[:2*n].reshape(n, 2)
                sides = result.x[2*n:]
                if _is_feasible(centers, sides):
                    s = _sum_sides(sides)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers.copy()
                        best_sides = sides.copy()
        except:
            pass
    
    # Final refinement of best solution
    if best_centers is not None and time.time() < deadline - 30:
        try:
            new_centers, new_sides = _refine(best_centers, best_sides)
            if _is_feasible(new_centers, new_sides):
                s = _sum_sides(new_sides)
                if s > best_sum:
                    best_sum = s
                    best_centers = new_centers.copy()
                    best_sides = new_sides.copy()
        except:
            pass
    
    # Ensure we have a valid result
    if best_centers is None:
        # Fallback to grid
        k = 3
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
