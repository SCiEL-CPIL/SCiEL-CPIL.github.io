# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, subject to non-overlapping interiors.
    
    Strategy:
    1. Variable transformation: sides = exp(log_sides) to ensure positivity without bounds.
    2. Smooth objective: Use softplus for constraint violations to provide continuous
       gradients that push squares apart, avoiding the "stuck in local minimum" issue
       of hard penalty functions.
    3. Multi-start with diverse topologies (grid, rows, random) to escape symmetry.
    4. Final polish: A high-precision local solve on the best feasible candidate found.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 610  # Leave 20s margin for safety

    # Softplus function: log(1 + exp(x)) / beta
    def softplus(x, beta=100.0):
        # Numerically stable softplus
        return np.logaddexp(0, x * beta) / beta

    def smooth_objective(x):
        """
        Minimize -sum(sides) + penalty.
        x = [cx_0, cy_0, ..., cx_9, cy_9, log_s_0, ..., log_s_9]
        """
        centers = x[:2 * n].reshape(n, 2)
        log_sides = x[2 * n:]
        sides = np.exp(log_sides)
        
        # 1. Maximize sum of sides (minimize negative sum)
        score = -np.sum(sides)
        
        # 2. Boundary penalties (smooth)
        # Left/Right/Bottom/Top violations
        pen_left = softplus(-(centers[:, 0] - sides / 2) + tol)
        pen_right = softplus((centers[:, 0] + sides / 2) - (1.0 + tol))
        pen_bottom = softplus(-(centers[:, 1] - sides / 2) + tol)
        pen_top = softplus((centers[:, 1] + sides / 2) - (1.0 + tol))
        
        boundary_pen = np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
        
        # 3. Overlap penalties (smooth)
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2 - tol
                
                # Violation if dx < t AND dy < t
                # We want to penalize when both are violated.
                # A smooth way: penalize the "amount" they overlap in both dimensions.
                # If dx < t, violation_x = t - dx > 0.
                # If dy < t, violation_y = t - dy > 0.
                # Overlap exists if both > 0.
                # We can penalize min(violation_x, violation_y) or a smooth approximation.
                # Actually, the constraint is |dx| >= t OR |dy| >= t.
                # Violation is when |dx| < t AND |dy| < t.
                # Let vx = t - dx, vy = t - dy.
                # If vx > 0 and vy > 0, we have overlap.
                # Penalty = softplus(vx) * softplus(vy) ? No, that's weak.
                # Better: Penalty = softplus(min(vx, vy)) ?
                # If they overlap by 0.1 in x and 0.1 in y, min is 0.1.
                # If they touch in x (vx=0) and overlap in y, min is 0. No penalty. Correct.
                # If they overlap in x by 0.01 and y by 0.01, min is 0.01. Small penalty.
                # This creates a gradient that pushes them apart along the axis of least separation.
                
                vx = t - dx
                vy = t - dy
                min_v = np.minimum(vx, vy)
                overlap_pen += softplus(min_v)
        
        # Weight the penalties heavily to ensure feasibility
        total_pen = 10.0 * (boundary_pen + overlap_pen)
        
        return score + total_pen

    def hard_feasibility(centers, sides):
        """Strict check for the final candidate."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    # Initial configurations
    starts = []
    
    # 1. Grid (4x3 or similar, but 10 is awkward for square grid, so 2x5 or 5x2)
    # Let's try a 2x5 grid first (2 rows, 5 cols)
    k1, k2 = 2, 5
    side1 = min(1.0/k1, 1.0/k2) # 0.2
    cx1, cy1 = np.meshgrid(np.linspace(side1/2, 1-side1/2, k2), np.linspace(side1/2, 1-side1/2, k1))
    centers1 = np.column_stack((cx1.flatten(), cy1.flatten()))
    sides1 = np.full(n, side1)
    log_sides1 = np.log(sides1)
    starts.append(np.concatenate([centers1.flatten(), log_sides1]))

    # 2. 5x2 grid
    k1, k2 = 5, 2
    side2 = min(1.0/k1, 1.0/k2) # 0.2
    cx2, cy2 = np.meshgrid(np.linspace(side2/2, 1-side2/2, k2), np.linspace(side2/2, 1-side2/2, k1))
    centers2 = np.column_stack((cx2.flatten(), cy2.flatten()))
    sides2 = np.full(n, side2)
    log_sides2 = np.log(sides2)
    starts.append(np.concatenate([centers2.flatten(), log_sides2]))

    # 3. Random feasible-ish
    rng = np.random.default_rng(42)
    for trial in range(5):
        sides_r = rng.uniform(0.15, 0.35, n)
        centers_r = np.zeros((n, 2))
        for i in range(n):
            s = sides_r[i]
            centers_r[i, 0] = rng.uniform(s/2, 1-s/2)
            centers_r[i, 1] = rng.uniform(s/2, 1-s/2)
        log_sides_r = np.log(sides_r)
        starts.append(np.concatenate([centers_r.flatten(), log_sides_r]))

    # 4. "Big and small" heuristic
    # One large square in corner, others around?
    # Just a random start with one large side
    sides_m = np.full(n, 0.2)
    sides_m[0] = 0.4
    centers_m = np.zeros((n, 2))
    centers_m[0] = [0.2, 0.2]
    for i in range(1, n):
        centers_m[i] = rng.uniform(0.2, 0.8, 2)
    log_sides_m = np.log(sides_m)
    starts.append(np.concatenate([centers_m.flatten(), log_sides_m]))

    best_x = None
    best_score = -np.inf
    best_feasible_x = None
    best_feasible_score = -np.inf

    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            # L-BFGS-B with bounds on centers [0,1] and log_sides (-inf, inf) effectively
            # But we should bound centers to [0,1] to help convergence.
            bounds = []
            for i in range(n):
                bounds.append((0.0, 1.0)) # cx
                bounds.append((0.0, 1.0)) # cy
            for i in range(n):
                bounds.append((-10.0, 1.0)) # log_sides (sides in [e^-10, e])
            
            res = minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x_new = res.x
            centers_new = x_new[:2*n].reshape(n, 2)
            sides_new = np.exp(x_new[2*n:])
            
            # Check if this is feasible in the hard sense
            if hard_feasibility(centers_new, sides_new):
                score = np.sum(sides_new)
                if score > best_feasible_score:
                    best_feasible_score = score
                    best_feasible_x = x_new
            
            # Also track the best raw score (might be infeasible but close)
            # But we only return feasible.
            
        except Exception:
            pass

    # If we have a feasible solution, try to polish it
    if best_feasible_x is not None:
        # Use the feasible solution as a starting point for a tighter optimization
        # We can use a very high beta in softplus to make it almost hard, 
        # or just run L-BFGS-B again with tight tolerances.
        
        # Let's try a few more iterations from the best feasible point
        x0_polish = best_feasible_x
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((-10.0, 1.0))
            
        try:
            res_polish = minimize(smooth_objective, x0_polish, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            x_polished = res_polish.x
            centers_polished = x_polished[:2*n].reshape(n, 2)
            sides_polished = np.exp(x_polished[2*n:])
            
            if hard_feasibility(centers_polished, sides_polished):
                score_polished = np.sum(sides_polished)
                if score_polished > best_feasible_score:
                    best_feasible_score = score_polished
                    best_feasible_x = x_polished
                    
        except Exception:
            pass

    # Fallback if no feasible solution found (should not happen with grid)
    if best_feasible_x is None:
        # Return the grid
        centers = np.column_stack((np.linspace(0.1, 0.9, 5).repeat(2), 
                                   np.tile(np.linspace(0.1, 0.9, 2), 5)))
        # Wait, the grid construction above was meshgrid. Let's just use the first start's result if it was feasible?
        # The first start is a grid of side 0.2.
        # Let's reconstruct a valid grid.
        k1, k2 = 2, 5
        side = 0.2
        cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, k2), np.linspace(side/2, 1-side/2, k1))
        centers = np.column_stack((cx.flatten(), cy.flatten()))
        sides = np.full(n, side)
        return centers, sides

    centers = best_feasible_x[:2*n].reshape(n, 2)
    sides = np.exp(best_feasible_x[2*n:])
    
    # Final safety check
    if not hard_feasibility(centers, sides):
         # This is bad, return grid
         k1, k2 = 2, 5
         side = 0.2
         cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, k2), np.linspace(side/2, 1-side/2, k1))
         centers = np.column_stack((cx.flatten(), cy.flatten()))
         sides = np.full(n, side)

    return centers, sides

# EVOLVE-BLOCK-END
