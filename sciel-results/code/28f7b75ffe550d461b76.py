"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5c
    # y = c * H
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    # Compute barycentric coordinates
    # c = 2*y/sqrt(3), b = x - 0.5*c, a = 1-b-c
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    # Distance to boundary is min(a, b, c) if negative
    # We want to penalize if any of a,b,c < 0
    # Penalty = sum(max(0, -a)^2 + max(0, -b)^2 + max(0, -c)^2)
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    # Clamp points for area calculation to avoid numerical issues with area
    pts_clamped = _clamp_all(pts)
    
    # Use a soft min area for smoothness
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    # Large penalty weight to keep points inside
    return -area_val + 100.0 * penalty

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    """
    alpha = x[0]
    return -alpha

def _nlp_constraint_func(x):
    """
    Returns array of (area_i - alpha) for all triangles.
    We want area_i >= alpha, so area_i - alpha >= 0.
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    areas = _all_areas(pts)
    return areas - alpha

def _nlp_jacobian(x):
    """
    Jacobian of the constraints with respect to x.
    x = [alpha, x1, y1, ..., x11, y11]
    Constraints: area_i - alpha >= 0 for all i
    
    d(area_i - alpha)/d(alpha) = -1
    d(area_i - alpha)/d(x_j) = d(area_i)/d(x_j)
    d(area_i - alpha)/d(y_j) = d(area_i)/d(y_j)
    
    Area of triangle (p_i, p_j, p_k) = 0.5 * |det([p_j - p_i, p_k - p_i])|
    Assuming non-degenerate and consistent orientation, 
    area = 0.5 * (x_j - x_i)(y_k - y_i) - 0.5 * (y_j - y_i)(x_k - x_i)
    (taking the signed area and assuming positive, or using abs which is non-diff)
    
    For a smooth objective, we need to be careful. 
    Let's use the formula: Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    
    Actually, let's just use finite differences or a simple gradient if needed, 
    but scipy SLSQP can use finite differences if jac is not provided.
    However, providing a Jacobian is much faster.
    
    Let's derive d(area)/d(x_m) and d(area)/d(y_m) for triangle (i, j, k).
    Let A = 0.5 * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j))
    dA/dx_i = 0.5 * (y_j - y_k)
    dA/dy_i = 0.5 * (x_k - x_j)
    dA/dx_j = 0.5 * (y_k - y_i)
    dA/dy_j = 0.5 * (x_i - x_k)
    dA/dx_k = 0.5 * (y_i - y_j)
    dA/dy_k = 0.5 * (x_j - x_i)
    
    Note: This is the SIGNED area. The actual area is |A|.
    If A > 0, d|A|/dx = dA/dx. If A < 0, d|A|/dx = -dA/dx.
    If A is close to 0, the gradient is undefined.
    For the NLP solver, we assume the points are in general position and A != 0.
    We can compute the sign of A for each triangle and adjust the gradient accordingly.
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    
    # Jacobian shape: (_N_TRI, 1 + 2*N)
    jac = np.zeros((_N_TRI, 1 + 2 * N))
    
    # d/d(alpha) = -1 for all constraints
    jac[:, 0] = -1.0
    
    for idx, (i, j, k) in enumerate(_TRI_IDX):
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        
        # Signed area * 2
        signed_area_2 = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
        
        # Gradient of signed area w.r.t coordinates
        # dA/dx_i = 0.5 * (y_j - y_k)
        gxi = 0.5 * (yj - yk)
        gyi = 0.5 * (xk - xj)
        gxj = 0.5 * (yk - yi)
        gyj = 0.5 * (xi - xk)
        gxk = 0.5 * (yi - yj)
        gyk = 0.5 * (xj - xi)
        
        # If signed area is negative, flip signs
        if signed_area_2 < 0:
            gxi, gyi, gxj, gyj, gxk, gyk = -gxi, -gyi, -gxj, -gyj, -gxk, -gyk
        
        # Map to x indices: x1..x11 are at indices 1, 3, 5...
        # y1..y11 are at indices 2, 4, 6...
        # Point m (0-indexed) has x at 2*m + 1, y at 2*m + 2
        jac[idx, 2*i + 1] = gxi
        jac[idx, 2*i + 2] = gyi
        jac[idx, 2*j + 1] = gxj
        jac[idx, 2*j + 2] = gyj
        jac[idx, 2*k + 1] = gxk
        jac[idx, 2*k + 2] = gyk
        
    return jac

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for differential evolution
    # x in [0, 1], y in [0, H]
    bounds_de = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # Run Differential Evolution to find a good starting point
    best_pts = None
    try:
        if time_left() > 30.0:
            res_de = differential_evolution(
                _objective_de,
                bounds_de,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                polish=False,
                updating='immediate'
            )
            if res_de.success or res_de.fun < 1e6:
                best_pts = _clamp_all(res_de.x.reshape(-1, 2))
            else:
                best_pts = _initial_placement_symmetric()
        else:
            best_pts = _initial_placement_symmetric()
    except Exception:
        best_pts = _initial_placement_symmetric()
    
    current_best = _min_area(best_pts)
    
    # Phase 1: NLP (SLSQP) to maximize min area directly
    # x = [alpha, x1, y1, ..., x11, y11]
    if time_left() > 20.0:
        try:
            initial_alpha = current_best
            x0 = np.concatenate(([initial_alpha], best_pts.flatten()))
            
            # Bounds for NLP
            bounds_nlp = [(0.0, 0.5)] + [(0.0, 1.0)] * N + [(0.0, H)] * N
            
            # Constraints: area_i - alpha >= 0
            # We need to provide the constraint function and optionally Jacobian
            # scipy SLSQP accepts 'type': 'ineq' which means f(x) >= 0
            
            res_nlp = minimize(
                _nlp_objective,
                x0,
                method='SLSQP',
                bounds=bounds_nlp,
                constraints=[
                    {'type': 'ineq', 'fun': _nlp_constraint_func, 'jac': _nlp_jacobian}
                ],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_nlp.success or res_nlp.fun < -initial_alpha:
                # res_nlp.x[0] is alpha, res_nlp.x[1:] are points
                nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                nlp_val = _min_area(nlp_pts)
                if nlp_val > current_best:
                    best_pts = nlp_pts
                    current_best = nlp_val
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B on soft objective for polishing
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=50.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > current_best:
                best_pts = polished_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
            
    # Phase 3: Another round of NLP if time permits, starting from the polished point
    if time_left() > 15.0:
        try:
            initial_alpha = current_best
            x0 = np.concatenate(([initial_alpha], best_pts.flatten()))
            
            bounds_nlp = [(0.0, 0.5)] + [(0.0, 1.0)] * N + [(0.0, H)] * N
            
            res_nlp2 = minimize(
                _nlp_objective,
                x0,
                method='SLSQP',
                bounds=bounds_nlp,
                constraints=[
                    {'type': 'ineq', 'fun': _nlp_constraint_func, 'jac': _nlp_jacobian}
                ],
                options={'maxiter': 100, 'ftol': 1e-14, 'disp': False}
            )
            
            if res_nlp2.success or res_nlp2.fun < -initial_alpha:
                nlp_pts = _clamp_all(res_nlp2.x[1:].reshape(-1, 2))
                nlp_val = _min_area(nlp_pts)
                if nlp_val > current_best:
                    best_pts = nlp_pts
                    current_best = nlp_val
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead polish on hard objective
    if time_left() > 5.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
