# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _get_critical_triangles(pts, k=10):
    """
    Identify the k-smallest area triangles.
    Returns list of (i, j, k) tuples for these triangles.
    """
    areas = _all_areas(pts)
    indices = np.argsort(areas)[:k]
    return _TRI_IDX[indices]

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    """
    alpha = x[0]
    return -alpha

def _nlp_constraints_critical(x, critical_tris):
    """
    Constraints for NLP: each critical triangle area >= alpha
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    
    n_crit = len(critical_tris)
    areas = np.zeros(n_crit)
    for idx, (i, j, k) in enumerate(critical_tris):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        dx1 = pj[0] - pi[0]
        dy1 = pj[1] - pi[1]
        dx2 = pk[0] - pi[0]
        dy2 = pk[1] - pi[1]
        areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
    
    return areas - alpha

def _nlp_jacobian_constraints_critical(x, critical_tris):
    """
    Jacobian of critical constraints w.r.t. x
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    n_vars = 1 + 2 * N
    n_crit = len(critical_tris)
    
    jac = np.zeros((n_crit, n_vars))
    jac[:, 0] = -1.0
    
    for idx, (i, j, k) in enumerate(critical_tris):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        sign = np.sign(cross)
        if sign == 0:
            sign = 1.0
        
        jac[idx, 1 + 2*i] = 0.5 * sign * (pj[1] - pk[1])
        jac[idx, 2 + 2*i] = 0.5 * sign * (pk[0] - pj[0])
        
        jac[idx, 1 + 2*j] = 0.5 * sign * (pi[1] - pk[1])
        jac[idx, 2 + 2*j] = 0.5 * sign * (pk[0] - pi[0])
        
        jac[idx, 1 + 2*k] = 0.5 * sign * (pj[1] - pi[1])
        jac[idx, 2 + 2*k] = 0.5 * sign * (pi[0] - pj[0])
    
    return jac

def _nlp_jacobian_objective(x):
    """Jacobian of objective"""
    n_vars = 1 + 2 * N
    jac = np.zeros(n_vars)
    jac[0] = -1.0
    return jac

def _nlp_jacobian_hessian_lagrange(x, v):
    """Hessian of Lagrangian"""
    n_vars = 1 + 2 * N
    return np.zeros((n_vars, n_vars))

def _nlp_bounds():
    """
    Bounds for NLP
    """
    bounds = [(0.0, 0.5)]
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
    return bounds

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    pts.append([1/3, H/3])
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    init_pts_list = []
    init_pts_list.append(_initial_placement_symmetric())
    
    for _ in range(4):
        random_pts = rng.uniform(0, 1, size=(N, 2))
        random_pts[:, 1] *= H
        random_pts = _clamp_all(random_pts)
        init_pts_list.append(random_pts)
    
    best_pts = init_pts_list[0]
    best_val = _min_area(best_pts)
    
    # Phase 1: Local Minimization from multiple starts
    for init_pts in init_pts_list:
        if time_left() < 10.0:
            break
        try:
            res = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                init_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            new_pts = _clamp_all(res.x.reshape(-1, 2))
            new_val = _min_area(new_pts)
            if new_val > best_val:
                best_val = new_val
                best_pts = new_pts
        except Exception:
            pass

    # Phase 2: NLP Refinement
    if time_left() > 15.0:
        try:
            critical_tris = _get_critical_triangles(best_pts, k=10)
            
            alpha_init = _min_area(best_pts) * 0.95
            x0 = np.zeros(1 + 2 * N)
            x0[0] = alpha_init
            x0[1:] = best_pts.flatten()
            
            nlp_bounds = _nlp_bounds()
            
            def _constraint_func(x, crit_tris=critical_tris):
                return _nlp_constraints_critical(x, crit_tris)
            
            def _constraint_jac(x, crit_tris=critical_tris):
                return _nlp_jacobian_constraints_critical(x, crit_tris)
            
            from scipy.optimize import NonlinearConstraint
            
            nlp_con = NonlinearConstraint(
                _constraint_func,
                0.0,
                np.inf,
                jac=_constraint_jac
            )
            
            res_nlp = minimize(
                _nlp_objective,
                x0,
                method='trust-constr',
                bounds=nlp_bounds,
                constraints=[nlp_con],
                jac=_nlp_jacobian_objective,
                hess=_nlp_jacobian_hessian_lagrange,
                options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            
            if res_nlp.success or (hasattr(res_nlp, 'fun') and res_nlp.fun < 0):
                nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                new_min = _min_area(nlp_pts)
                if new_min > best_val:
                    best_pts = nlp_pts
                    best_val = new_min
        except Exception:
            pass

    # Phase 3: Final Polish with Powell
    if time_left() > 5.0:
        try:
            bounds_powell = [(0.0, 1.0), (0.0, H)] * N
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=bounds_powell,
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            final_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            if _min_area(final_pts) > best_val:
                best_pts = final_pts
                best_val = _min_area(best_pts)
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
