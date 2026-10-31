"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

# EVOLVE-BLOCK-START

def _check_validity(circles, tol=1e-10):
    """Check if circles are valid (disjoint, inside unit square)."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _make_initial_grid():
    """Create a reasonable initial grid layout."""
    # Use a hexagonal-like packing pattern
    # 5 columns x 5 rows = 25, plus 1 extra
    # Or 4x4=16 + 10 extra... let's try a better approach
    
    # Start with a 5x5 grid (25 circles) + 1
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    # Add one more in a corner-ish spot
    positions.append([0.5, 0.5])  # center, will be pushed
    
    # Actually, let's use a more structured initial guess
    # 26 circles: try 5x5 grid with one removed and repositioned
    # Or just start with equal circles in a grid
    
    # Better: use a hexagonal packing pattern
    # For 26 circles, let's use a pattern close to optimal
    # Start with 5 columns, varying rows
    
    # Simple approach: 5x5 grid + 1, then optimize
    positions = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            positions.append([x, y])
    positions.append([0.5, 0.5])  # 26th circle
    
    # Initial radii: small enough to not overlap
    # In a 5x5 grid, spacing is 0.2, so radius ~0.09
    r_init = 0.09
    circles = np.array(positions + [[r_init]] * len(positions))
    # Wait, that's wrong format. Let me fix.
    
    circles = np.zeros((len(positions), 3))
    for i, pos in enumerate(positions):
        circles[i, 0] = pos[0]
        circles[i, 1] = pos[1]
        circles[i, 2] = r_init
    
    return circles

def _objective_and_constraints(circles):
    """
    Objective: maximize sum of radii (so minimize negative sum)
    Constraints:
    - Circles inside unit square: x >= r, x <= 1-r, y >= r, y <= 1-r
    - Non-overlapping: dist(i,j) >= r_i + r_j
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    
    # Inequality constraints (scipy expects g(x) >= 0)
    # Inside square:
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    
    constraints = []
    
    # Boundary constraints
    constraints.append({'type': 'ineq', 'fun': lambda c: c[0] - c[2]})  # x - r >= 0
    constraints.append({'type': 'ineq', 'fun': lambda c: 1 - c[0] - c[2]})  # 1-x-r >= 0
    constraints.append({'type': 'ineq', 'fun': lambda c: c[1] - c[2]})  # y - r >= 0
    constraints.append({'type': 'ineq', 'fun': lambda c: 1 - c[1] - c[2]})  # 1-y-r >= 0
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            def _nonoverlap(c, i=i, j=j):
                xi, yi, ri = c[i*3], c[i*3+1], c[i*3+2]
                xj, yj, rj = c[j*3], c[j*3+1], c[j*3+2]
                return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
            constraints.append({'type': 'ineq', 'fun': _nonoverlap})
    
    return obj, constraints

def _solve_slsqp(circles_init, maxiter=500, ftol=1e-8):
    """Solve using SLSQP."""
    n = len(circles_init)
    x0 = circles_init.flatten()
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    # Objective function
    def obj(x):
        c = x.reshape(n, 3)
        return -np.sum(c[:, 2])
    
    # Constraints
    cons = []
    
    # Boundary: x >= r
    def bound_x_min(x):
        c = x.reshape(n, 3)
        return c[:, 0] - c[:, 2]
    cons.append({'type': 'ineq', 'fun': bound_x_min})
    
    # Boundary: 1-x >= r
    def bound_x_max(x):
        c = x.reshape(n, 3)
        return 1 - c[:, 0] - c[:, 2]
    cons.append({'type': 'ineq', 'fun': bound_x_max})
    
    # Boundary: y >= r
    def bound_y_min(x):
        c = x.reshape(n, 3)
        return c[:, 1] - c[:, 2]
    cons.append({'type': 'ineq', 'fun': bound_y_min})
    
    # Boundary: 1-y >= r
    def bound_y_max(x):
        c = x.reshape(n, 3)
        return 1 - c[:, 1] - c[:, 2]
    cons.append({'type': 'ineq', 'fun': bound_y_max})
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            def nonoverlap(x, i=i, j=j):
                c = x.reshape(n, 3)
                xi, yi, ri = c[i, 0], c[i, 1], c[i, 2]
                xj, yj, rj = c[j, 0], c[j, 1], c[j, 2]
                return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
            cons.append({'type': 'ineq', 'fun': nonoverlap})
    
    result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': maxiter, 'ftol': ftol})
    
    if result.success:
        return result.x.reshape(n, 3)
    else:
        return circles_init

def _penalty_method(circles_init, maxiter=100, penalty_start=10.0, penalty_mult=10.0):
    """Penalty method for non-smooth constraints."""
    n = len(circles_init)
    x = circles_init.flatten().copy()
    penalty = penalty_start
    
    for it in range(maxiter):
        def obj_with_penalty(x):
            c = x.reshape(n, 3)
            x_c = c[:, 0]
            y_c = c[:, 1]
            r_c = c[:, 2]
            
            # Main objective
            obj = -np.sum(r_c)
            
            # Penalty terms
            pen = 0.0
            
            # Boundary violations
            pen += np.sum(np.maximum(0, r_c - x_c)**2)
            pen += np.sum(np.maximum(0, r_c - (1 - x_c))**2)
            pen += np.sum(np.maximum(0, r_c - y_c)**2)
            pen += np.sum(np.maximum(0, r_c - (1 - y_c))**2)
            
            # Pairwise overlap
            for i in range(n):
                for j in range(i+1, n):
                    dist_sq = (x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2
                    sum_r = r_c[i] + r_c[j]
                    violation = max(0, sum_r - np.sqrt(max(dist_sq, 1e-12)))
                    pen += violation**2
            
            return obj + penalty * pen
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-6, 1-1e-6))  # x
            bounds.append((1e-6, 1-1e-6))  # y
            bounds.append((1e-6, 0.5))  # r
        
        result = minimize(obj_with_penalty, x, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-10})
        
        if result.success:
            new_x = result.x
            # Check if constraints are satisfied
            c = new_x.reshape(n, 3)
            x_c = c[:, 0]
            y_c = c[:, 1]
            r_c = c[:, 2]
            
            max_viol = 0.0
            max_viol = max(max_viol, np.max(np.maximum(0, r_c - x_c)))
            max_viol = max(max_viol, np.max(np.maximum(0, r_c - (1 - x_c))))
            max_viol = max(max_viol, np.max(np.maximum(0, r_c - y_c)))
            max_viol = max(max_viol, np.max(np.maximum(0, r_c - (1 - y_c))))
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2)
                    viol = max(0, r_c[i] + r_c[j] - dist)
                    max_viol = max(max_viol, viol)
            
            if max_viol < 1e-8:
                return new_x.reshape(n, 3)
            
            x = new_x
        else:
            break
        
        penalty *= penalty_mult
    
    return x.reshape(n, 3)

def _final_refinement(circles):
    """Multiple SLSQP passes with decreasing ftol."""
    n = len(circles)
    result = circles.copy()
    
    for ftol in [1e-8, 1e-10, 1e-12]:
        try:
            new_result = _solve_slsqp(result, maxiter=200, ftol=ftol)
            if _sum_radii(new_result) >= _sum_radii(result) - 1e-10:
                result = new_result
        except Exception:
            pass
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initial guess
    circles = _make_initial_grid()
    best = circles.copy()
    best_score = _sum_radii(best)
    
    # Try SLSQP first
    if time.time() < deadline - 30:
        try:
            result = _solve_slsqp(circles, maxiter=500, ftol=1e-8)
            if _sum_radii(result) > best_score:
                best = result
                best_score = _sum_radii(best)
        except Exception:
            pass
    
    # Try penalty method
    if time.time() < deadline - 30:
        try:
            result = _penalty_method(circles, maxiter=50, penalty_start=10.0)
            if _sum_radii(result) > best_score:
                best = result
                best_score = _sum_radii(best)
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 10:
        try:
            best = _final_refinement(best)
        except Exception:
            pass
    
    # Validate and shrink if needed
    if not _check_validity(best, tol=1e-8):
        # Shrink radii slightly
        best = best.copy()
        best[:, 2] *= 0.99
        if not _check_validity(best, tol=1e-8):
            # More aggressive shrink
            best[:, 2] *= 0.95
    
    return best

# EVOLVE-BLOCK-END
