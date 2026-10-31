"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _check_validity(circles):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _objective_and_gradient(circles, N):
    """Objective: negative sum of radii. Gradient computed analytically."""
    circles = np.array(circles).reshape(N, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Constraints:
    # 1. Circles inside unit square: x >= r, x <= 1-r, y >= r, y <= 1-r
    # 2. Disjoint interiors: dist(i,j) >= ri + rj
    
    # We use a penalty method for simplicity in gradient computation
    # But let's use a proper constrained optimization approach
    
    # Actually, let's reformulate: maximize sum(r) subject to constraints
    # Using SLSQP with constraints
    
    # For the objective function for SLSQP: minimize -sum(r)
    return -np.sum(r)

def _construct_initial_grid(N):
    """Create initial grid arrangement."""
    # Try different grid configurations
    best_arrangement = None
    best_sum = -1
    
    for cols in range(1, 10):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
        
        # Calculate radius for this grid
        # Circles centered at (i+0.5)/cols, (j+0.5)/rows
        # Radius limited by min(0.5/cols, 0.5/rows) but also by spacing
        r = min(0.5 / cols, 0.5 / rows) - 0.001
        
        if r <= 0:
            continue
            
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) < N:
                    cx = (i + 0.5) / cols
                    cy = (j + 0.5) / rows
                    circles.append([cx, cy, r])
        
        if len(circles) == N:
            total = sum(c[2] for c in circles)
            if total > best_sum:
                best_sum = total
                best_arrangement = circles
    
    if best_arrangement is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        r = 0.5 / side - 0.001
        best_arrangement = [[(i + 0.5) / side, (j + 0.5) / side, r] 
                          for i in range(side) for j in range(side)][:N]
    
    return best_arrangement

def _optimize_circles(circles_init, N, time_budget=100.0):
    """Optimize circle positions and radii using SLSQP."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Flatten circles
    x0 = np.array(circles_init).flatten()
    
    # Define constraints
    def _constraints(circles_flat):
        c = circles_flat.reshape(N, 3)
        x, y, r = c[:, 0], c[:, 1], c[:, 2]
        
        cons = []
        
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2], 'jac': lambda c, i=i: np.array([1.0 if k == 3*i else 0.0 if k != 3*i+2 else -1.0 for k in range(3*N)])})
            cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1.0 - c[3*i] - c[3*i+2], 'jac': lambda c, i=i: np.array([-1.0 if k == 3*i else 0.0 if k != 3*i+2 else -1.0 for k in range(3*N)])})
            cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2], 'jac': lambda c, i=i: np.array([0.0 if k != 3*i+1 else 1.0 if k != 3*i+2 else -1.0 for k in range(3*N)])})
            cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1.0 - c[3*i+1] - c[3*i+2], 'jac': lambda c, i=i: np.array([0.0 if k != 3*i+1 else -1.0 if k != 3*i+2 else -1.0 for k in range(3*N)])})
        
        # Disjointness constraints: dist(i,j) >= ri + rj
        for i in range(N):
            for j in range(i + 1, N):
                def _dist_constraint(c, i=i, j=j):
                    xi, yi, ri = c[3*i], c[3*i+1], c[3*i+2]
                    xj, yj, rj = c[3*j], c[3*j+1], c[3*j+2]
                    d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    return d - ri - rj
                
                def _dist_jacobian(c, i=i, j=j):
                    xi, yi, ri = c[3*i], c[3*i+1], c[3*i+2]
                    xj, yj, rj = c[3*j], c[3*j+1], c[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx**2 + dy**2)
                    if d < 1e-12:
                        d = 1e-12
                    jac = np.zeros(3*N)
                    jac[3*i] = dx/d
                    jac[3*i+1] = dy/d
                    jac[3*i+2] = -1.0
                    jac[3*j] = -dx/d
                    jac[3*j+1] = -dy/d
                    jac[3*j+2] = -1.0
                    return jac
                
                cons.append({'type': 'ineq', 'fun': _dist_constraint, 'jac': _dist_jacobian})
        
        return cons
    
    # This constraint formulation is too expensive. Let's use a penalty method instead.
    
    def _penalty_objective(circles_flat, penalty_weight=100.0):
        c = circles_flat.reshape(N, 3)
        x, y, r = c[:, 0], c[:, 1], c[:, 2]
        
        obj = -np.sum(r)
        
        # Boundary violations
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                obj += penalty_weight * (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                obj += penalty_weight * (y[i] - (1 - r[i]))**2
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    obj += penalty_weight * (r[i] + r[j] - d)**2
        
        return obj
    
    # Use Nelder-Mead for simplicity and robustness
    best_circles = np.array(circles_init).copy()
    best_obj = _penalty_objective(best_circles.flatten())
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial solution
        if restart > 0:
            x0 = best_circles.flatten() + np.random.normal(0, 0.01, 3*N)
        else:
            x0 = np.array(circles_init).flatten()
        
        # Clip to valid ranges
        for i in range(N):
            x0[3*i+2] = max(0.001, min(0.5, x0[3*i+2]))  # radius
            x0[3*i] = max(0, min(1, x0[3*i]))  # x
            x0[3*i+1] = max(0, min(1, x0[3*i+1]))  # y
        
        try:
            result = minimize(_penalty_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_circles = result.x.reshape(N, 3)
        except Exception:
            continue
        
        if time.time() > deadline - 5:
            break
    
    # Project to valid solution
    circles = best_circles.copy()
    for i in range(N):
        # Ensure inside unit square
        r = circles[i, 2]
        circles[i, 0] = max(r, min(1 - r, circles[i, 0]))
        circles[i, 1] = max(r, min(1 - r, circles[i, 1]))
        circles[i, 2] = max(0.001, min(0.5, circles[i, 2]))
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, leave margin
    
    # Try multiple initial configurations and optimize
    best_result = None
    best_score = -1
    
    # Initial grid arrangements
    initial_arrangements = [_construct_initial_grid(N)]
    
    # Add some perturbed versions
    base = initial_arrangements[0]
    for _ in range(3):
        perturbed = []
        for c in base:
            x = c[0] + random.gauss(0, 0.02)
            y = c[1] + random.gauss(0, 0.02)
            r = c[2] * (1 + random.gauss(0, 0.05))
            r = max(0.01, min(0.4, r))
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            perturbed.append([x, y, r])
        initial_arrangements.append(perturbed)
    
    for arr in initial_arrangements:
        if time.time() > deadline - 20:
            break
        
        try:
            optimized = _optimize_circles(arr, N, time_budget=min(20, deadline - time.time() - 5))
            if _check_validity(optimized):
                score = sum(c[2] for c in optimized)
                if score > best_score:
                    best_score = score
                    best_result = optimized
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to initial grid
        best_result = np.array(initial_arrangements[0])
    
    return np.array(best_result)
# EVOLVE-BLOCK-END
