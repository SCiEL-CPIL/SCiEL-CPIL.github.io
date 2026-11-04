"""
Strategy: Successive Convex Relaxation (SCR) with L-BFGS-B Polish for Circle Packing

This approach uses SCR to find a good feasible packing, then polishes with
direct nonlinear optimization for higher precision.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _check_validity(circles, w, h, tol=1e-6):
    """Check if circles are valid: within bounds and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > w + tol:
            return False
        if y - r < -tol or y + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _separate_circles(circles, w, h, max_iter=500, tol=1e-8):
    """
    Repair invalid circles by iteratively separating overlapping pairs
    and pushing circles inside bounds.
    Returns (repaired_circles, valid).
    """
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_iter):
        moved = False
        
        # Push circles inside bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            elif x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            elif y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        # Separate overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist:
                    if dist < 1e-12:
                        # Circles at same position; push apart
                        circles[i, 0] += min_dist / 2
                        circles[j, 0] -= min_dist / 2
                        moved = True
                    else:
                        overlap = min_dist - dist
                        # Push apart along the connecting line
                        dx_unit = dx / dist
                        dy_unit = dy / dist
                        circles[i, 0] += overlap / 2 * dx_unit
                        circles[i, 1] += overlap / 2 * dy_unit
                        circles[j, 0] -= overlap / 2 * dx_unit
                        circles[j, 1] -= overlap / 2 * dy_unit
                        moved = True
        
        # Re-check bounds after separation
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            elif x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            elif y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        if not moved:
            break
    
    if _check_validity(circles, w, h, tol=1e-6):
        return circles, True
    else:
        # Final attempt: shrink radii slightly
        for scale in [0.995, 0.99, 0.98, 0.97, 0.95]:
            circles_shrunk = circles.copy()
            circles_shrunk[:, 2] *= scale
            if _check_validity(circles_shrunk, w, h, tol=1e-6):
                return circles_shrunk, True
        return circles, False

def _polish_constrained(circles, w, h, max_iter=1000, ftol=1e-12):
    """
    Polish circles using SLSQP with strict non-overlap and boundary constraints.
    This avoids penalty weight tuning issues.
    """
    n = len(circles)
    x0 = circles.flatten()
    
    # Objective: Maximize sum of radii -> Minimize -sum(r)
    def objective(x):
        return -np.sum(x[2::3])
    
    # Gradient of objective
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return grad
    
    # Constraints
    constraints = []
    
    # 1. Non-overlap constraints: dist(i,j) >= r_i + r_j
    # SLSQP handles inequality constraints g(x) >= 0
    # We define g(x) = dist(i,j) - r_i - r_j
    # Note: SLSQP requires fun and jac for each constraint.
    # For 21 circles, there are 210 pairs. This might be slow for SLSQP.
    # Alternative: Use a single function returning an array of constraint values?
    # scipy.optimize.minimize with SLSQP supports 'type': 'ineq' with a vector-valued fun.
    
    def constraint_fun(x):
        cons = np.zeros((n*(n-1)//2,))
        idx = 0
        x_coords = x[0::3]
        y_coords = x[1::3]
        r = x[2::3]
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons[idx] = dist - r[i] - r[j]
                idx += 1
        return cons
    
    def constraint_jac(x):
        # Jacobian of constraint_fun
        # Shape: (n_pairs, 3n)
        n_pairs = n*(n-1)//2
        J = np.zeros((n_pairs, 3*n))
        idx = 0
        x_coords = x[0::3]
        y_coords = x[1::3]
        r = x[2::3]
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-8:
                    # Degenerate case, provide arbitrary gradient to help escape
                    # d/dx_i (dist - r_i - r_j) approx 1, d/dx_j approx -1
                    J[idx, 3*i] = 1.0
                    J[idx, 3*j] = -1.0
                    J[idx, 3*i+2] = -1.0
                    J[idx, 3*j+2] = -1.0
                else:
                    ux = dx / dist
                    uy = dy / dist
                    # d(dist)/dx_i = ux, d(dist)/dy_i = uy
                    # d(dist)/dx_j = -ux, d(dist)/dy_j = -uy
                    # d(-r_i)/dr_i = -1, d(-r_j)/dr_j = -1
                    J[idx, 3*i] = ux
                    J[idx, 3*i+1] = uy
                    J[idx, 3*i+2] = -1.0
                    J[idx, 3*j] = -ux
                    J[idx, 3*j+1] = -uy
                    J[idx, 3*j+2] = -1.0
                idx += 1
        return J
    
    constraints.append({'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac})
    
    # 2. Boundary constraints:
    # x_i - r_i >= 0
    # w - x_i - r_i >= 0
    # y_i - r_i >= 0
    # h - y_i - r_i >= 0
    
    def boundary_fun(x):
        cons = np.zeros((4*n,))
        x_coords = x[0::3]
        y_coords = x[1::3]
        r = x[2::3]
        for i in range(n):
            cons[4*i] = x_coords[i] - r[i]       # Left
            cons[4*i+1] = w - x_coords[i] - r[i] # Right
            cons[4*i+2] = y_coords[i] - r[i]     # Bottom
            cons[4*i+3] = h - y_coords[i] - r[i] # Top
        return cons
    
    def boundary_jac(x):
        J = np.zeros((4*n, 3*n))
        for i in range(n):
            # Left: x_i - r_i
            J[4*i, 3*i] = 1.0
            J[4*i, 3*i+2] = -1.0
            # Right: w - x_i - r_i
            J[4*i+1, 3*i] = -1.0
            J[4*i+1, 3*i+2] = -1.0
            # Bottom: y_i - r_i
            J[4*i+2, 3*i+1] = 1.0
            J[4*i+2, 3*i+2] = -1.0
            # Top: h - y_i - r_i
            J[4*i+3, 3*i+1] = -1.0
            J[4*i+3, 3*i+2] = -1.0
        return J
    
    constraints.append({'type': 'ineq', 'fun': boundary_fun, 'jac': boundary_jac})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, 1.0))
    
    res = minimize(
        objective,
        x0,
        jac=objective_grad,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': ftol}
    )
    
    return res.x.reshape(n, 3)

def _polish(circles, w, h):
    """
    Full polish: separate, then constrained optimize, then check validity.
    """
    # First, try to repair any violations
    circles_fixed, valid = _separate_circles(circles, w, h, max_iter=200, tol=1e-7)
    
    if valid:
        # Polish with constrained optimization
        circles_polished = _polish_constrained(circles_fixed, w, h, max_iter=500, ftol=1e-12)
        
        # Check validity of polished result
        if _check_validity(circles_polished, w, h, tol=1e-6):
            # Final separation pass to ensure validity (tighten if needed)
            circles_final, valid_final = _separate_circles(circles_polished, w, h, max_iter=100, tol=1e-8)
            if valid_final:
                return circles_final
            else:
                # If separation fails, try shrinking slightly
                for scale in [0.999, 0.995, 0.99]:
                    circles_shrunk = circles_polished.copy()
                    circles_shrunk[:, 2] *= scale
                    if _check_validity(circles_shrunk, w, h, tol=1e-6):
                        return circles_shrunk
        else:
            # If polish made it invalid, use the separated version
            return circles_fixed
    else:
        # If separation failed, try with smaller radii
        for scale in [0.99, 0.98, 0.97, 0.95]:
            circles_shrunk = circles.copy()
            circles_shrunk[:, 2] *= scale
            circles_shrunk_fixed, valid_shrunk = _separate_circles(circles_shrunk, w, h, max_iter=200, tol=1e-7)
            if valid_shrunk:
                circles_polished = _polish_constrained(circles_shrunk_fixed, w, h, max_iter=200, ftol=1e-10)
                if _check_validity(circles_polished, w, h, tol=1e-6):
                    circles_final, valid_final = _separate_circles(circles_polished, w, h, max_iter=100, tol=1e-8)
                    if valid_final:
                        return circles_final
                return circles_shrunk_fixed
        return circles
    
    return circles

def _grid_initialization(w, h, n_circles):
    """Create an initial grid placement of circles."""
    side = int(np.ceil(np.sqrt(n_circles)))
    circles = np.zeros((n_circles, 3))
    for i in range(n_circles):
        row = i // side
        col = i % side
        circles[i, 0] = (col + 0.5) * (w / side)
        circles[i, 1] = (row + 0.5) * (h / side)
        min_dim = min(w, h)
        circles[i, 2] = 0.5 * min_dim / side * 0.3  # Small initial radius
    return circles

def _random_initialization(w, h, n_circles, seed=42):
    """Create a random initial placement of circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n_circles, 3))
    min_dim = min(w, h)
    for i in range(n_circles):
        circles[i, 0] = rng.uniform(0.05, w - 0.05)
        circles[i, 1] = rng.uniform(0.05, h - 0.05)
        circles[i, 2] = rng.uniform(0.02, 0.08)
    return circles

def _successive_convex_relaxation(circles, w, h, max_iter=30):
    """
    SCR: Iteratively linearize non-overlap constraints and solve LP-like subproblem.
    circles: (n, 3) array
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    for _ in range(max_iter):
        # Build linearized constraints
        # Variables: x = [dx_0, dy_0, ..., dx_{N-1}, dy_{N-1}, r_0, ..., r_{N-1}]
        n_vars = 3 * n
        
        def objective(x):
            r = x[2 * n:3 * n]
            return -np.sum(r)
        
        def constraint_func(x):
            p = x[:2 * n].reshape(n, 2)
            r = x[2 * n:3 * n]
            cons = []
            
            # Non-overlap linearized constraints
            for i in range(n):
                for j in range(i + 1, n):
                    dy = pos[i] - pos[j]
                    norm_y = np.linalg.norm(dy)
                    if norm_y < 1e-8:
                        dy = np.array([1.0, 0.0])
                        norm_y = 1.0
                    else:
                        dy = dy / norm_y
                    
                    val = np.dot(dy, p[i] - p[j]) - r[i] - r[j]
                    cons.append(val)
            
            # Wall constraints
            for i in range(n):
                cons.append(p[i, 0] - r[i])          # Left
                cons.append(w - p[i, 0] - r[i])      # Right
                cons.append(p[i, 1] - r[i])          # Bottom
                cons.append(h - p[i, 1] - r[i])      # Top
            
            return cons
        
        constraints = [{'type': 'ineq', 'fun': constraint_func}]
        
        bounds = []
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, 1.0))
        
        x0 = np.concatenate([pos.flatten(), radii])
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-12})
        
        if res.success:
            p_new = res.x[:2 * n].reshape(n, 2)
            r_new = res.x[2 * n:3 * n]
            
            if np.sum(r_new) <= np.sum(radii) + 1e-8:
                break
            
            # Check validity
            circles_new = np.column_stack([p_new, r_new])
            if _check_validity(circles_new, w, h, tol=1e-5):
                pos = p_new
                radii = r_new
            else:
                # Try to repair
                circles_repaired, valid = _separate_circles(circles_new, w, h, max_iter=100, tol=1e-6)
                if valid:
                    pos = circles_repaired[:, :2]
                    radii = circles_repaired[:, 2]
                else:
                    break
        else:
            break
    
    return np.column_stack([pos, radii])

def _best_known():
    """
    Return a known good packing for 21 circles in a 1x1 square (perimeter 4).
    This is a fallback; the actual optimization should do better.
    """
    # Simple 5x5 grid with one extra circle
    w, h = 1.0, 1.0
    side = 5
    circles = np.zeros((N, 3))
    r = 0.08
    for i in range(N):
        row = i // side
        col = i % side
        circles[i, 0] = (col + 0.5) / side
        circles[i, 1] = (row + 0.5) / side
        circles[i, 2] = r
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # Focus on ratios near 1.0 which are often optimal for circle packing in rectangles
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.05, 0.95, 1.15, 0.8, 1.25]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        circles = _grid_initialization(w, h, N)
        
        # SCR
        circles_scr = _successive_convex_relaxation(circles, w, h, max_iter=30)
        
        # Check if we have time for polishing
        if time.time() > deadline - 15.0:
            # Not enough time for polish; just use SCR result
            if _check_validity(circles_scr, w, h, tol=1e-6):
                value = np.sum(circles_scr[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = circles_scr.copy()
            continue
        
        # Polish
        circles_polished = _polish(circles_scr, w, h)
        
        # Final validation
        if _check_validity(circles_polished, w, h, tol=1e-6):
            value = np.sum(circles_polished[:, 2])
            if value > best_value:
                best_value = value
                best_result = circles_polished.copy()
        else:
            # Try repair
            circles_repaired, valid = _separate_circles(circles_polished, w, h, max_iter=200, tol=1e-6)
            if valid:
                value = np.sum(circles_repaired[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = circles_repaired.copy()
    
    # If we still have time, try a few random restarts with different seeds
    if time.time() < deadline - 20.0:
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Vary aspect ratio slightly
            aspect = 1.0 + 0.1 * (seed - 2)
            w = 2.0 / (1.0 + aspect)
            h = 2.0 - w
            
            circles = _random_initialization(w, h, N, seed=seed + 100)
            circles_scr = _successive_convex_relaxation(circles, w, h, max_iter=20)
            
            if time.time() < deadline - 10.0:
                circles_polished = _polish(circles_scr, w, h)
                if _check_validity(circles_polished, w, h, tol=1e-6):
                    value = np.sum(circles_polished[:, 2])
                    if value > best_value:
                        best_value = value
                        best_result = circles_polished.copy()
    
    if best_result is None:
        # Fallback
        best_result = _best_known()
    
    return best_result
