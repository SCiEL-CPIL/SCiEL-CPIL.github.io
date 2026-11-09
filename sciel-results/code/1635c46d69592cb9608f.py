import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    
    Strategy:
    1. Define a smooth penalty-based objective that penalizes boundary violations and overlaps.
    2. Use Differential Evolution (global search) to navigate the continuous landscape.
    3. Use multiple diverse starting configurations and local refinement (Nelder-Mead) to escape local optima.
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def objective(x):
        """
        x: [s_0, ..., s_9, cx_0, ..., cx_9, cy_0, ..., cy_9]
        Returns: -sum(s) + penalty
        """
        s = x[:n]
        cx = x[n:2 * n]
        cy = x[2 * n:3 * n]
        
        # Ensure sides are positive (clip for penalty calc, but let optimizer handle)
        s_pos = np.maximum(s, 1e-9)
        
        penalty = 0.0
        M = 1e6  # Large penalty weight
        
        # Boundary penalties
        for i in range(n):
            # Left
            if cx[i] - s_pos[i] / 2 < 0:
                penalty += M * (cx[i] - s_pos[i] / 2) ** 2
            # Right
            if cx[i] + s_pos[i] / 2 > 1:
                penalty += M * (cx[i] + s_pos[i] / 2 - 1) ** 2
            # Bottom
            if cy[i] - s_pos[i] / 2 < 0:
                penalty += M * (cy[i] - s_pos[i] / 2) ** 2
            # Top
            if cy[i] + s_pos[i] / 2 > 1:
                penalty += M * (cy[i] + s_pos[i] / 2 - 1) ** 2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sep_x = (s_pos[i] + s_pos[j]) / 2
                sep_y = (s_pos[i] + s_pos[j]) / 2
                
                # If overlapping in both x and y, penalty is proportional to intrusion
                # Smooth approximation: max(0, sep_x - dx) * max(0, sep_y - dy)
                pen_x = max(0, sep_x - dx)
                pen_y = max(0, sep_y - dy)
                penalty += M * pen_x * pen_y
                
        return -np.sum(s) + penalty

    # Differential Evolution bounds
    bounds = [(0, 1)] * (3 * n)

    best_val = 0.0
    best_x = None

    # Helper to convert x to centers/sides and check validity
    def x_to_cs(x):
        s = x[:n]
        cx = x[n:2 * n]
        cy = x[2 * n:3 * n]
        centers = np.column_stack([cx, cy])
        return centers, s

    # Initial best guess: 3x3 grid + 1 (approx 3.33)
    # Let's start with a reasonable feasible solution
    # 4x3 grid? No, 10 squares. 
    # Simple grid: 3 cols, 4 rows. Side 1/3? No, 1/4 in y? 
    # Let's just use a 1/3 side grid for 9 squares, and one small one?
    # Or just let DE find it. But we need a valid fallback.
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx9 = (cols + 0.5) * side
    cy9 = (rows + 0.5) * side
    # 10th square: small, in a gap? Or just make all 10 side 0.3?
    # If we use side 0.3, we can fit 3x3=9. 10th must be smaller or we need better packing.
    # Fallback: 10 squares of side 0.25 in a 4x3 grid (4 cols, 3 rows = 12 slots).
    # Side 0.25. Sum = 2.5. This is worse than 3.33.
    # Better fallback: 3x3 grid of 1/3, and 10th square side 0.1 in a corner?
    # Actually, just return the best found by DE. If DE fails, return a valid grid.
    
    # Valid Grid Fallback: 4 cols, 3 rows. Side 0.25.
    # Centers: x = 0.125, 0.375, 0.625, 0.875
    #          y = 0.125, 0.375, 0.625
    # 10 squares: fill 4x3 grid (12 spots), pick 10.
    # Let's pick the first 10 in row-major order.
    cx_fallback = []
    cy_fallback = []
    sides_fallback = []
    count = 0
    for r in range(3):
        for c in range(4):
            if count < 10:
                cx_fallback.append((c + 0.5) * 0.25)
                cy_fallback.append((r + 0.5) * 0.25)
                sides_fallback.append(0.25)
                count += 1
    centers_fallback = np.array(cx_fallback, dtype=np.float64).reshape(10, 1)
    cy_fallback_arr = np.array(cy_fallback, dtype=np.float64).reshape(10, 1)
    centers_fallback = np.hstack([centers_fallback, cy_fallback_arr])
    sides_fallback = np.array(sides_fallback, dtype=np.float64)
    
    if is_valid(centers_fallback, sides_fallback):
        best_val = np.sum(sides_fallback)
        best_x = np.concatenate([sides_fallback, centers_fallback[:, 0], centers_fallback[:, 1]])
    else:
        # Should not happen, but if it does, use tiny squares
        best_val = 0.1
        best_x = np.zeros(30)
        best_x[:10] = 0.01
        # Place them far apart
        for i in range(10):
            best_x[10+i] = 0.1 + 0.08*i
            best_x[20+i] = 0.1 + 0.08*i

    # Run Differential Evolution
    # DE is stochastic, so we can run it multiple times or with different seeds
    # We have ~580s. DE with 30 vars, 45 constraints (implicit).
    # Let's run DE with a reasonable maxiter and popsize.
    
    # To improve diversity, we can run DE multiple times with different seeds
    # or use a large population.
    
    # Strategy: Run DE with popsize=50, maxiter=500. 
    # If time permits, run again with different seed.
    
    def run_de(seed):
        nonlocal best_val, best_x
        if time.time() > deadline - 30:
            return
        
        try:
            # Use a callback to track best
            def callback(xk, convergence=True):
                nonlocal best_val, best_x
                # xk is the current population best? No, DE callback gets the best individual in current gen
                # Actually, scipy DE callback gets xk (best in pop), convergence flag.
                val = -objective(xk)
                # Check validity strictly
                centers, s = x_to_cs(xk)
                if is_valid(centers, s):
                    if val > best_val:
                        best_val = val
                        best_x = xk.copy()
                return False  # Don't stop early
            
            res = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=300,
                popsize=30,
                tol=1e-8,
                callback=callback,
                polish=True,  # Use L-BFGS-B or Nelder-Mead at the end? 
                # polish uses L-BFGS-B by default in some versions, or Nelder-Mead.
                # L-BFGS-B doesn't handle non-smooth penalties well. 
                # Let's disable polish in DE and do our own Nelder-Mead.
                polish=False 
            )
            
            # Check the result
            val = -res.fun
            centers, s = x_to_cs(res.x)
            if is_valid(centers, s):
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
            
            # Local refinement on best found
            if time.time() < deadline - 10:
                res2 = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                val2 = -res2.fun
                centers2, s2 = x_to_cs(res2.x)
                if is_valid(centers2, s2):
                    if val2 > best_val:
                        best_val = val2
                        best_x = res2.x.copy()

        except Exception as e:
            pass

    # Run DE with a few different seeds to explore different basins
    seeds = [42, 123, 456]
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        run_de(seed)
        
        # If we have time, do another refinement pass
        if time.time() < deadline - 30:
             # Perturb best_x slightly and run Nelder-Mead again
             np.random.seed(seed + 100)
             perturbed_x = best_x + np.random.normal(0, 0.01, 30)
             # Clip to bounds
             perturbed_x = np.clip(perturbed_x, 0, 1)
             try:
                 res3 = minimize(
                     objective,
                     perturbed_x,
                     method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                 )
                 val3 = -res3.fun
                 centers3, s3 = x_to_cs(res3.x)
                 if is_valid(centers3, s3):
                     if val3 > best_val:
                         best_val = val3
                         best_x = res3.x.copy()
             except:
                 pass

    # Final extraction
    if best_x is None:
        centers, sides = centers_fallback, sides_fallback
    else:
        centers, sides = x_to_cs(best_x)
        
        # Final validity check
        if not is_valid(centers, sides):
            centers, sides = centers_fallback, sides_fallback

    return centers, sides
