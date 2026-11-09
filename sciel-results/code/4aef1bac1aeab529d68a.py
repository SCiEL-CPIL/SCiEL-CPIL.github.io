# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearProgramming


def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 580

    # Known good starting points
    # 1. 4x3 grid (with one empty spot) - actually 4x3 = 12, we need 10
    # Let's use a 4x4 grid but only 10 squares, or better specific packings.
    
    # Strategy: Use a set of diverse initial configurations
    # 1. Grid 4x3 (12 slots, pick 10)
    # 2. Grid 5x2
    # 3. Grid 2x5
    # 4. Random
    # 5. Specific known good packings if any (e.g., 5 big squares?)
    
    # A very simple strong baseline: 
    # 5 squares of side 0.4? No, 5*0.4^2 = 0.8 area, fits. 
    # But sum of sides = 2.0.
    # 10 squares of side 0.3? 10*0.09 = 0.9 area. Sum = 3.0.
    # 10 squares of side 0.31? 10*0.0961 = 0.961 area. Sum = 3.1.
    
    # Let's generate a good grid-based initial guess.
    # A 4x4 grid of 0.25 squares has sum 2.5.
    # We can do better.
    
    # Let's try to place 10 squares in a 4x3 arrangement but with varying sizes?
    # Or simply start with a dense grid and let the optimizer work.
    
    # Better initial guess:
    # Divide unit square into 4 columns and 3 rows. 
    # 12 cells. We need 10. 
    # If we use 10 cells of a 4x3 grid (size 0.25 x 0.333), the squares would be min(0.25, 0.333) = 0.25.
    # Sum = 2.5.
    
    # Let's try 5 columns, 2 rows. 10 cells exactly.
    # Width 0.2, Height 0.5. Square side = 0.2.
    # Sum = 10 * 0.2 = 2.0.
    
    # Let's try 2 columns, 5 rows. Same.
    
    # What about a mix?
    # Let's just use a randomized restart approach with SLSQP.
    
    best_cx = np.zeros(n)
    best_cy = np.zeros(n)
    best_s = np.zeros(n)
    best_obj = 0.0
    
    # Initial guess 1: 4x3 grid (using first 10 of 12 spots)
    # Actually, let's use a 4x4 grid and pick 10 spots that are well distributed.
    # Or just a 5x2 grid.
    kx, ky = 5, 2
    idx = np.arange(n)
    cols = idx % kx
    rows = idx // kx
    cx0 = (cols + 0.5) * (1.0 / kx)
    cy0 = (rows + 0.5) * (1.0 / ky)
    s0 = np.full(n, min(1.0/kx, 1.0/ky))
    
    if np.sum(s0) > best_obj:
        best_cx = cx0.copy()
        best_cy = cy0.copy()
        best_s = s0.copy()
        best_obj = np.sum(s0)
        
    # Initial guess 2: 4x3 grid (using 10 of 12 spots, maybe skip 2 corners?)
    # Let's just use a 4x3 grid where we place squares in a 4x3 array but 
    # since 12 > 10, we can make some squares larger if we remove neighbors.
    # This is complex. Let's stick to simple grids and random.
    
    # Initial guess 3: 3x4 grid
    kx, ky = 3, 4
    idx = np.arange(n)
    cols = idx % kx
    rows = idx // kx
    cx0 = (cols + 0.5) * (1.0 / kx)
    cy0 = (rows + 0.5) * (1.0 / ky)
    s0 = np.full(n, min(1.0/kx, 1.0/ky))
    if np.sum(s0) > best_obj:
        best_cx = cx0.copy()
        best_cy = cy0.copy()
        best_s = s0.copy()
        best_obj = np.sum(s0)
        
    # Initial guess 4: Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        s = rng.uniform(0.1, 0.3, n)
        # Clip to bounds
        for i in range(n):
            s[i] = min(s[i], 0.4)
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
        if np.sum(s) > best_obj:
             # Check if feasible? If not, shrink until feasible.
             # For initial best, we just want a decent start for SLSQP.
             # We will run SLSQP on this and see if it converges to something valid.
             pass

    # We will run SLSQP from multiple starting points.
    # To make SLSQP work, we need to define separation constraints.
    # The separation pattern (which side touches which) is discrete.
    # We can infer the pattern from the initial guess.
    
    def run_slsqp(cx_init, cy_init, s_init):
        # Determine separation pattern
        # For each pair (i, j), decide if i is left/right of j or below/above j.
        # We choose the axis with larger separation.
        sep_pattern = {}
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx_init[i] - cx_init[j])
                dy = abs(cy_init[i] - cy_init[j])
                if dx >= dy:
                    if cx_init[i] < cx_init[j]:
                        sep_pattern[(i, j)] = 0  # i left of j
                    else:
                        sep_pattern[(i, j)] = 1  # i right of j
                else:
                    if cy_init[i] < cy_init[j]:
                        sep_pattern[(i, j)] = 2  # i below j
                    else:
                        sep_pattern[(i, j)] = 3  # i above j

        # Variables: cx[0..n-1], cy[0..n-1], s[0..n-1]
        x0 = np.concatenate([cx_init, cy_init, s_init])
        
        def objective(x):
            return -np.sum(x[2*n:])
            
        def objective_grad(x):
            g = np.zeros(3*n)
            g[2*n:] = -np.ones(n)
            return g
            
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            # cx_i >= s_i/2
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - x[2*n+i]/2,
                'jac': lambda x, i=i: np.concatenate([
                    np.array([1.0 if k==i else 0.0 for k in range(n)]),
                    np.zeros(n),
                    np.array([-0.5 if k==i else 0.0 for k in range(n)])
                ])
            })
            # cx_i <= 1 - s_i/2
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[i] - x[2*n+i]/2,
                'jac': lambda x, i=i: np.concatenate([
                    np.array([-1.0 if k==i else 0.0 for k in range(n)]),
                    np.zeros(n),
                    np.array([-0.5 if k==i else 0.0 for k in range(n)])
                ])
            })
            # cy_i >= s_i/2
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2,
                'jac': lambda x, i=i: np.concatenate([
                    np.zeros(n),
                    np.array([1.0 if k==i else 0.0 for k in range(n)]),
                    np.array([-0.5 if k==i else 0.0 for k in range(n)])
                ])
            })
            # cy_i <= 1 - s_i/2
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]/2,
                'jac': lambda x, i=i: np.concatenate([
                    np.zeros(n),
                    np.array([-1.0 if k==i else 0.0 for k in range(n)]),
                    np.array([-0.5 if k==i else 0.0 for k in range(n)])
                ])
            })
            # s_i >= 0.01
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[2*n+i] - 0.01,
                'jac': lambda x, i=i: np.concatenate([
                    np.zeros(2*n),
                    np.array([1.0 if k==i else 0.0 for k in range(n)])
                ])
            })
            
        # Separation constraints
        for (i, j), pat in sep_pattern.items():
            if pat == 0: # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[j] - x[i] - x[2*n+i]/2 - x[2*n+j]/2,
                    'jac': lambda x, i=i, j=j: np.concatenate([
                        np.array([-1.0 if k==i else (1.0 if k==j else 0.0) for k in range(n)]),
                        np.zeros(n),
                        np.array([-0.5 if k==i else (-0.5 if k==j else 0.0) for k in range(n)])
                    ])
                })
            elif pat == 1: # i right of j: cx_j + s_j/2 <= cx_i - s_i/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[i] - x[j] - x[2*n+i]/2 - x[2*n+j]/2,
                    'jac': lambda x, i=i, j=j: np.concatenate([
                        np.array([1.0 if k==i else (-1.0 if k==j else 0.0) for k in range(n)]),
                        np.zeros(n),
                        np.array([-0.5 if k==i else (-0.5 if k==j else 0.0) for k in range(n)])
                    ])
                })
            elif pat == 2: # i below j: cy_i + s_i/2 <= cy_j - s_j/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[n+j] - x[n+i] - x[2*n+i]/2 - x[2*n+j]/2,
                    'jac': lambda x, i=i, j=j: np.concatenate([
                        np.zeros(n),
                        np.array([-1.0 if k==i else (1.0 if k==j else 0.0) for k in range(n)]),
                        np.array([-0.5 if k==i else (-0.5 if k==j else 0.0) for k in range(n)])
                    ])
                })
            elif pat == 3: # i above j: cy_j + s_j/2 <= cy_i - s_i/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[n+i] - x[n+j] - x[2*n+i]/2 - x[2*n+j]/2,
                    'jac': lambda x, i=i, j=j: np.concatenate([
                        np.zeros(n),
                        np.array([1.0 if k==i else (-1.0 if k==j else 0.0) for k in range(n)]),
                        np.array([-0.5 if k==i else (-0.5 if k==j else 0.0) for k in range(n)])
                    ])
                })
                
        bounds = [(0, 1)] * (2*n) + [(0.01, 1)] * n
        
        try:
            result = minimize(
                objective, x0, jac=objective_grad,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.success:
                x_sol = result.x
                cx_sol = x_sol[:n]
                cy_sol = x_sol[n:2*n]
                s_sol = x_sol[2*n:]
                # Validate
                if _valid(np.stack([cx_sol, cy_sol], axis=1), s_sol):
                    return cx_sol, cy_sol, s_sol, np.sum(s_sol)
        except Exception:
            pass
        return None

    # Run SLSQP from the initial best
    res = run_slsqp(best_cx, best_cy, best_s)
    if res:
        best_cx, best_cy, best_s, best_obj = res

    # Try random restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        s = rng.uniform(0.2, 0.4, n)
        
        # Make it feasible by shrinking
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
            
        # Check feasibility, if not, shrink largest squares
        # Simple check
        feasible = True
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2
                if abs(cx[i]-cx[j]) < t and abs(cy[i]-cy[j]) < t:
                    # Overlap, shrink s[i] and s[j]
                    s[i] *= 0.8
                    s[j] *= 0.8
                    cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
                    cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
                    cx[j] = np.clip(cx[j], s[j]/2, 1 - s[j]/2)
                    cy[j] = np.clip(cy[j], s[j]/2, 1 - s[j]/2)
                    feasible = False
        if not feasible:
            # Still might be infeasible, let's just try to run SLSQP
            # SLSQP might fix it if the separation pattern is reasonable
            pass
            
        res = run_slsqp(cx, cy, s)
        if res:
            cx_sol, cy_sol, s_sol, obj_sol = res
            if obj_sol > best_obj:
                best_cx, best_cy, best_s, best_obj = cx_sol, cy_sol, s_sol, obj_sol

    # Final polish: run SLSQP again from the best found
    if time.time() < deadline - 30:
        res = run_slsqp(best_cx, best_cy, best_s)
        if res:
            cx_sol, cy_sol, s_sol, obj_sol = res
            if obj_sol > best_obj:
                best_cx, best_cy, best_s, best_obj = cx_sol, cy_sol, s_sol, obj_sol

    # Ensure validity one last time
    if not _valid(np.stack([best_cx, best_cy], axis=1), best_s):
        # Fallback to a very safe configuration
        # 10 squares of side 0.3 in a grid?
        # 5x2 grid
        kx, ky = 5, 2
        idx = np.arange(n)
        cols = idx % kx
        rows = idx // kx
        best_cx = (cols + 0.5) * (1.0 / kx)
        best_cy = (rows + 0.5) * (1.0 / ky)
        best_s = np.full(n, min(1.0/kx, 1.0/ky))

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides


def _valid(centers, sides, tol=1e-7):
    """Check validity of a packing configuration."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

# EVOLVE-BLOCK-END
