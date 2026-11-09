import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    # Initial heuristic layout: 3 rows
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.875],
        [0.875, 0.875]
    ])
    sides_init = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])

    def to_flat(centers, sides):
        x = np.zeros(3 * n)
        for i in range(n):
            x[3*i] = centers[i, 0]
            x[3*i+1] = centers[i, 1]
            x[3*i+2] = sides[i]
        return x

    def from_flat(x):
        cx = x[0::3]
        cy = x[1::3]
        s = np.maximum(x[2::3], 1e-8)
        centers = np.stack([cx, cy], axis=1)
        return centers, s

    def validate(centers, sides):
        """Check if configuration is valid; return (valid, score)."""
        tol = 1e-7
        s = np.maximum(sides, 1e-8)
        for i in range(n):
            if centers[i, 0] - s[i]/2 < -tol or centers[i, 0] + s[i]/2 > 1 + tol:
                return False, 0.0
            if centers[i, 1] - s[i]/2 < -tol or centers[i, 1] + s[i]/2 > 1 + tol:
                return False, 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (s[i] + s[j])/2 - tol
                if dx < t and dy < t:
                    return False, 0.0
        return True, np.sum(s)

    def clamp_to_box(centers, sides):
        """Clamp centers so squares fit in [0,1]^2."""
        s = np.maximum(sides, 1e-8)
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
        return np.stack([cx, cy], axis=1), s

    # SLSQP with smooth inequality constraints
    def objective(x):
        """Minimize negative sum of sides."""
        s = x[2::3]
        return -np.sum(np.maximum(s, 1e-10))

    def gradient_obj(x):
        s = x[2::3]
        grad = np.zeros(3*n)
        for i in range(n):
            grad[3*i+2] = -1.0 if s[i] > 1e-10 else 0.0
        return grad

    def make_constraints():
        """Create smooth inequality constraints for SLSQP.
        For each pair (i,j), we need: NOT(|dx| < req AND |dy| < req)
        i.e., |dx| >= req OR |dy| >= req
        We use the smooth constraint: min(|dx|-req, |dy|-req) >= 0
        Approximated via soft-min: -logsumexp(-a, -b)/alpha where a=|dx|-req, b=|dy|-req
        But simpler: use squared overlap as penalty-free constraint.
        
        Actually for SLSQP, we need g(x) >= 0.
        Use: g_ij = max(0, req - |dx|) * max(0, req - |dy|) should be 0
        But that's not differentiable. Instead use:
        g_ij = (req - |dx|)^2 + (req - |dy|)^2 - (req - |dx|)^2*(req - |dy|)^2/(req^2) ... complex.
        
        Simpler approach: use the product form as a smooth surrogate.
        g_ij = req^2 - (req - dx)^2 - (req - dy)^2 >= 0  (not quite right)
        
        Let me use: for each pair, two constraints:
        c1: dx >= req - eps  (separation in x)
        c2: dy >= req - eps  (separation in y)
        But we need OR, not AND.
        
        Best approach: use the complementarity-like smooth constraint:
        g_ij = (req - |dx|) * (req - |dy|) <= 0  means no overlap
        For SLSQP inequality: g_ij <= 0, i.e., -(req-|dx|)*(req-|dy|) >= 0
        This is smooth except at |dx|=req or |dy|=req, but SLSQP handles it.
        """
        cons = []
        # Boundary constraints: s_i/2 <= cx_i <= 1-s_i/2
        for i in range(n):
            # cx_i - s_i/2 >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([1.0, 0.0, -0.5])})
            # 1 - cx_i - s_i/2 >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([-1.0, 0.0, -0.5])})
            # cy_i - s_i/2 >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([0.0, 1.0, -0.5])})
            # 1 - cy_i - s_i/2 >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([0.0, -1.0, -0.5])})
            # s_i > 0
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i+2],
                         'jac': lambda x, i=i: np.array([0.0, 0.0, 1.0])})
        
        # Overlap constraints: for each pair, -(req - |dx|)*(req - |dy|) >= 0
        # i.e., (req - |dx|)*(req - |dy|) <= 0
        for i in range(n):
            for j in range(i+1, n):
                def overlap_con(x, i=i, j=j):
                    dx = abs(x[3*i] - x[3*j])
                    dy = abs(x[3*i+1] - x[3*j+1])
                    req = (x[3*i+2] + x[3*j+2]) / 2
                    # We want: NOT(dx < req AND dy < req)
                    # Equivalent: (req - dx) <= 0 OR (req - dy) <= 0
                    # Smooth: -(max(0, req-dx)) * (max(0, req-dy)) >= 0
                    # But max is not smooth. Use:
                    # g = (req - dx)*(req - dy) <= 0
                    # For SLSQP ineq: -(req-dx)*(req-dy) >= 0
                    return -(req - dx) * (req - dy)
                
                def overlap_jac(x, i=i, j=j):
                    dx = abs(x[3*i] - x[3*j])
                    dy = abs(x[3*i+1] - x[3*j+1])
                    req = (x[3*i+2] + x[3*j+2]) / 2
                    sign_dx = 1.0 if x[3*i] > x[3*j] else -1.0
                    sign_dy = 1.0 if x[3*i+1] > x[3*j+1] else -1.0
                    
                    grad = np.zeros(3*n)
                    # d/dx_i of -(req-dx)*(req-dy)
                    # = (req-dy)*sign_dx - (req-dx)*0.5 (from req)
                    # Actually: let a = req - dx, b = req - dy
                    # g = -a*b
                    # dg/dx_i = -[da/dx_i * b + a * db/dx_i]
                    # da/dx_i = -sign_dx - 0.5 (since req = (s_i+s_j)/2, dreq/ds_i = 0.5, dreq/dx_i = 0)
                    # Wait: req depends on s_i and s_j, not on x_i or y_i directly
                    # da/dx_i = -sign_dx (dx changes, req doesn't depend on x)
                    # da/ds_i = 0.5
                    # db/dx_i = 0
                    # db/ds_i = 0.5
                    # dg/dx_i = -[(-sign_dx)*b + a*0] = sign_dx * b
                    # dg/dy_i = -[0 + a*(-sign_dy)] = a*sign_dy
                    # dg/ds_i = -[0.5*b + a*0.5] = -0.5*(a+b)
                    
                    a = req - dx
                    b = req - dy
                    
                    grad[3*i] = sign_dx * b
                    grad[3*i+1] = sign_dy * a
                    grad[3*i+2] = -0.5 * (a + b)
                    
                    grad[3*j] = -sign_dx * b
                    grad[3*j+1] = -sign_dy * a
                    grad[3*j+2] = -0.5 * (a + b)
                    
                    return grad
                
                cons.append({'type': 'ineq',
                             'fun': overlap_con,
                             'jac': overlap_jac})
        return cons

    constraints = make_constraints()
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-8, 1.0)] * n

    def slsqp_solve(x0, max_iter=500):
        """Run SLSQP optimization."""
        try:
            res = minimize(objective, x0, jac=gradient_obj, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': max_iter, 'ftol': 1e-8})
            return res.x
        except Exception:
            return x0

    # Multi-start with SLSQP
    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    valid, best_score = validate(best_centers, best_sides)
    if not valid:
        best_centers, best_sides = clamp_to_box(best_centers, best_sides)
        valid, best_score = validate(best_centers, best_sides)

    # Generate diverse initial layouts
    layouts = []
    layouts.append((centers_init, sides_init))
    
    # 2x5 grid
    c2 = np.array([[0.1,0.25],[0.3,0.25],[0.5,0.25],[0.7,0.25],[0.9,0.25],
                    [0.1,0.75],[0.3,0.75],[0.5,0.75],[0.7,0.75],[0.9,0.75]])
    s2 = np.full(n, 0.2)
    layouts.append((c2, s2))
    
    # 5x2 grid
    c5 = np.array([[0.25,0.1],[0.25,0.3],[0.25,0.5],[0.25,0.7],[0.25,0.9],
                    [0.75,0.1],[0.75,0.3],[0.75,0.5],[0.75,0.7],[0.75,0.9]])
    s5 = np.full(n, 0.2)
    layouts.append((c5, s5))
    
    # Random layouts
    for _ in range(5):
        cr = np.random.uniform(0.1, 0.9, (n, 2))
        sr = np.random.uniform(0.08, 0.3, n)
        layouts.append((cr, sr))

    for c_init, s_init in layouts:
        if time.time() > deadline - 15:
            break
        c_fix, s_fix = clamp_to_box(c_init, s_init)
        x_flat = to_flat(c_fix, s_fix)
        x_res = slsqp_solve(x_flat, max_iter=300)
        centers_res, sides_res = from_flat(x_res)
        centers_res, sides_res = clamp_to_box(centers_res, sides_res)
        valid, score = validate(centers_res, sides_res)
        if valid and score > best_score:
            best_score = score
            best_centers = centers_res.copy()
            best_sides = sides_res.copy()

    # Final high-precision polish with more iterations
    if time.time() < deadline - 30:
        x_flat = to_flat(best_centers, best_sides)
        x_res = slsqp_solve(x_flat, max_iter=1000)
        centers_res, sides_res = from_flat(x_res)
        centers_res, sides_res = clamp_to_box(centers_res, sides_res)
        valid, score = validate(centers_res, sides_res)
        if valid and score > best_score:
            best_score = score
            best_centers = centers_res.copy()
            best_sides = sides_res.copy()

    # Second polish pass
    if time.time() < deadline - 15:
        x_flat = to_flat(best_centers, best_sides)
        x_res = slsqp_solve(x_flat, max_iter=500)
        centers_res, sides_res = from_flat(x_res)
        centers_res, sides_res = clamp_to_box(centers_res, sides_res)
        valid, score = validate(centers_res, sides_res)
        if valid and score > best_score:
            best_score = score
            best_centers = centers_res.copy()
            best_sides = sides_res.copy()

    # Final validation
    valid, final_score = validate(best_centers, best_sides)
    if not valid:
        return centers_init, sides_init

    return best_centers, best_sides
