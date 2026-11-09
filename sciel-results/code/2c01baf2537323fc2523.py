# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping squares in a unit square.
    Uses differential evolution with a penalty function for overlaps and boundary violations,
    followed by local refinement.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    # Variables: x_0, y_0, s_0, x_1, y_1, s_1, ...
    # Bounds:
    # x, y in [0, 1]
    # s in [0, 1]
    # Note: Center x must be in [s/2, 1-s/2], but we handle this with penalty.
    # For DE, we can bound x,y in [0,1] and s in [0,1].
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # s

    def decode(params):
        centers = np.empty((n, 2))
        sides = np.empty(n)
        for i in range(n):
            centers[i, 0] = params[3*i]
            centers[i, 1] = params[3*i+1]
            sides[i] = params[3*i+2]
        return centers, sides

    def penalty(params):
        centers, sides = decode(params)
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Boundary penalty
        pen = 0.0
        # Left
        pen += np.sum(np.maximum(0, s/2 - cx)**2)
        # Right
        pen += np.sum(np.maximum(0, (cx + s/2) - 1)**2)
        # Bottom
        pen += np.sum(np.maximum(0, s/2 - cy)**2)
        # Top
        pen += np.sum(np.maximum(0, (cy + s/2) - 1)**2)
        
        # Overlap penalty
        # For each pair, check if they overlap in both x and y
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2.0
                req_y = (s[i] + s[j]) / 2.0
                overlap_x = max(0, req_x - dx)
                overlap_y = max(0, req_y - dy)
                # If they overlap in both, penalize the product or sum?
                # Sum of squared overlaps is a standard smooth approximation.
                pen += overlap_x**2 + overlap_y**2
        
        return pen

    def objective(params):
        centers, sides = decode(params)
        total_side = np.sum(sides)
        p = penalty(params)
        # Weight for penalty. Large enough to prevent obvious overlaps, 
        # but small enough to allow exploration.
        # We want to maximize total_side, so minimize -total_side + lambda * penalty
        return -total_side + 100.0 * p

    # Initial guesses
    # 1. Grid 3x3 (10 cells used)
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    params_grid = np.empty(3*n)
    for i in range(n):
        params_grid[3*i] = cx[i]
        params_grid[3*i+1] = cy[i]
        params_grid[3*i+2] = side
        
    # 2. Random feasible-ish
    np.random.seed(42)
    # Generate random non-overlapping squares via rejection sampling is slow.
    # Just use random points and small sides for diversity.
    params_rand = np.random.uniform(0.1, 0.9, 3*n)
    # Force small sides to start feasible
    params_rand[2::3] = np.random.uniform(0.05, 0.2, n)

    # 3. Corner-based
    # 4 corners size 0.3
    # 1 center size 0.4
    # 5 edges size 0.2?
    params_corner = np.zeros(3*n)
    # Corners
    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    for i, (x, y) in enumerate(corners):
        params_corner[3*i] = x
        params_corner[3*i+1] = y
        params_corner[3*i+2] = 0.4
    # Center
    params_corner[3*4] = 0.5
    params_corner[3*4+1] = 0.5
    params_corner[3*4+2] = 0.4
    # Edges (midpoints of sides)
    edges = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5), (0.5, 0.5)] # Last one overlaps center, adjust
    # Let's just place 5 small squares in remaining space roughly
    # This seed is just for diversity, DE will fix it.
    for i in range(5):
        x, y = edges[i]
        params_corner[3*(5+i)] = x
        params_corner[3*(5+i)+1] = y
        params_corner[3*(5+i)+2] = 0.15

    best_params = params_grid
    best_score = objective(params_grid)
    
    seeds = [params_grid, params_rand, params_corner]
    
    # Run Differential Evolution
    # Maximize time budget
    # popsize=15, tol=1e-6
    # We can run multiple DE instances or just one with a large population.
    # Let's run one main DE instance.
    
    try:
        # DE is stochastic. We use a fixed seed for reproducibility but allow randomness.
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-6, 
            popsize=20,
            polish=True,  # Use L-BFGS-B to polish
            init='sohal' # Scaled Orthogonal Helical Array
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception as e:
        pass # Fallback to seeds if DE fails

    # Local Refinement with SLSQP
    # SLSQP can be more precise. We define constraints explicitly for boundaries.
    # Overlaps are still soft in the objective.
    
    def slsqp_objective(params):
        centers, sides = decode(params)
        return -np.sum(sides) + 100.0 * penalty(params)

    def slsqp_grad(params):
        # Numerical gradient is fine for 30 vars
        eps = 1e-8
        grad = np.zeros_like(params)
        f0 = slsqp_objective(params)
        for i in range(len(params)):
            params_p = params.copy()
            params_p[i] += eps
            grad[i] = (slsqp_objective(params_p) - f0) / eps
            params_p = params.copy()
        return grad

    # Boundary constraints (Hard)
    # cx - s/2 >= 0  => s/2 - cx <= 0
    # cx + s/2 <= 1  => cx + s/2 - 1 <= 0
    # cy - s/2 >= 0  => s/2 - cy <= 0
    # cy + s/2 <= 1  => cy + s/2 - 1 <= 0
    
    def make_bound_constraint(axis):
        # axis 0: x, 1: y
        def constraint(params):
            centers, sides = decode(params)
            c = centers[:, axis]
            s = sides
            # Left/Bottom: s/2 - c <= 0
            cons1 = s/2.0 - c
            # Right/Top: c + s/2 - 1 <= 0
            cons2 = c + s/2.0 - 1.0
            return np.concatenate([cons1, cons2])
        return constraint

    cons = []
    cons.append({'type': 'ineq', 'fun': make_bound_constraint(0)})
    cons.append({'type': 'ineq', 'fun': make_bound_constraint(1)})
    # s > 0
    def s_pos(params):
        centers, sides = decode(params)
        return sides
    cons.append({'type': 'ineq', 'fun': s_pos})

    # Try polishing the best result found so far
    try:
        res_slsqp = minimize(
            slsqp_objective, 
            best_params, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        if res_slsqp.fun < best_score:
            best_score = res_slsqp.fun
            best_params = res_slsqp.x
    except Exception:
        pass

    # If time remains, try another DE run from a different seed or with higher popsize
    if time.time() < deadline - 100:
        try:
            res2 = differential_evolution(
                objective, 
                bounds, 
                seed=123, 
                maxiter=100, 
                tol=1e-5, 
                popsize=30,
                polish=False
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
                
                # Polish again
                res_slsqp2 = minimize(
                    slsqp_objective, 
                    best_params, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 50, 'ftol': 1e-9}
                )
                if res_slsqp2.fun < best_score:
                    best_score = res_slsqp2.fun
                    best_params = res_slsqp2.x
        except Exception:
            pass

    centers, sides = decode(best_params)
    
    # Final validation and clipping to ensure strict validity within tol
    # The evaluator uses tol=1e-7.
    # We should ensure no square violates boundary by more than tol.
    # And no overlap.
    # If the optimizer left slight violations, we might need to shrink squares slightly.
    
    # Check validity
    def is_valid(c, s, tol=1e-7):
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        # Boundary
        if np.any(c[:,0] - s/2 < -tol) or np.any(c[:,0] + s/2 > 1+tol):
            return False
        if np.any(c[:,1] - s/2 < -tol) or np.any(c[:,1] + s/2 > 1+tol):
            return False
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(c[i,0] - c[j,0]) < t and abs(c[i,1] - c[j,1]) < t:
                    return False
        return True

    if not is_valid(centers, sides):
        # Fallback to grid if optimization failed validity
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        # Optional: Tiny shrink to be safe against floating point errors in evaluator
        # The evaluator allows tol=1e-7.
        # We can shrink sides by 1e-8 to be safe.
        sides = sides * (1 - 1e-8)
        # Re-check validity after shrink
        if not is_valid(centers, sides):
             # If shrink broke it (unlikely), revert to grid
             k = 3
             side = 1.0 / k
             idx = np.arange(n)
             cols = idx % k
             rows = idx // k
             cx = (cols + 0.5) * side
             cy = (rows + 0.5) * side
             centers = np.stack([cx, cy], axis=1).astype(np.float64)
             sides = np.full(n, side, dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
