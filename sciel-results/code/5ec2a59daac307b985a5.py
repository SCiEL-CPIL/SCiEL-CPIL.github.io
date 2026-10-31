"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices as a numpy array for vectorized ops
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all triangle areas using matrix indexing."""
    i_idx = _TRI_IDX[:, 0]
    j_idx = _TRI_IDX[:, 1]
    k_idx = _TRI_IDX[:, 2]
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    # Barycentric coordinates relative to V0, V1, V2
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Using the standard formula for equilateral triangle
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    eps = 1e-12
    a = max(eps, min(1.0 - 2*eps, a))
    b = max(eps, min(1.0 - 2*eps, b))
    c = max(eps, min(1.0 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _initial_placement_symmetric():
    """Symmetric initial placement using barycentric coordinates."""
    pts = []
    margin = 0.08
    pts.append(margin * V0 + (1-margin)/2 * V1 + (1-margin)/2 * V2)
    pts.append((1-margin)/2 * V0 + margin * V1 + (1-margin)/2 * V2)
    pts.append((1-margin)/2 * V0 + (1-margin)/2 * V1 + margin * V2)
    m2 = 0.05
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    pts.append((V0 + V1 + V2) / 3)
    cx, cy = 1/3, H/3
    r = 0.12
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _nlp_objective(x):
    """NLP objective: minimize -alpha (i.e., maximize alpha).
    x = [p1x, p1y, ..., p11x, p11y, alpha]
    """
    pts = x[:2*N].reshape(N, 2)
    alpha = x[2*N]
    return -alpha

def _nlp_constraints(x):
    """NLP constraints: area of every triangle >= alpha.
    Returns list of constraint dicts.
    """
    pts = x[:2*N].reshape(N, 2)
    alpha = x[2*N]
    areas = _all_areas(pts)
    # Each constraint: area_t - alpha >= 0
    # For SLSQP, we pass as a single array constraint
    return areas - alpha

def _nlp_jacobian_constraints(x):
    """Jacobian of constraints w.r.t. x.
    Constraint t: area_t(pts) - alpha >= 0
    d(area_t)/d(pts) and d(area_t)/d(alpha) = -1
    """
    pts = x[:2*N].reshape(N, 2)
    # Compute Jacobian of each area w.r.t. each point coordinate
    # area_t = 0.5 * |cross(pj-pi, pk-pi)|
    # We use the signed version and take abs, but for gradient we need to be careful.
    # For SLSQP, we'll use a numerical approach or simplified gradient.
    # Actually, let's use the signed cross product and handle the absolute value.
    # For the constraint, we can use the signed area (without abs) since we want
    # all areas to be positive (points in convex position within triangle).
    # But to be safe, let's compute gradients numerically via finite differences
    # if analytical is too complex.
    
    # Actually, let's compute analytical gradients for the signed area.
    # signed_area_t = 0.5 * (dx1*dy2 - dy1*dx2) where dx1=pjx-pix, dy1=pjy-piy, etc.
    # d(signed_area)/d(pix) = -0.5*(dy2 - dy1) ... this gets complex.
    # Let's use a simpler approach: for each triangle, the gradient of the
    # (signed) area w.r.t. each of the 3 points is well-known.
    
    # For triangle (i,j,k), signed area = 0.5*(xj-xi)*(yk-yi) - 0.5*(yj-yi)*(xk-xi)
    # d/d(xi) = -0.5*(yk-yi) + 0.5*(yj-yi) = 0.5*(yj-yk)
    # d/d(xj) = 0.5*(yk-yi)
    # d/d(xk) = -0.5*(yj-yi)
    # d/d(yi) = -0.5*(xk-xi) + 0.5*(xj-xi) = 0.5*(xj-xk)
    # d/d(yj) = 0.5*(xk-xi)
    # d/d(yk) = -0.5*(xj-xi)
    
    # Build the full Jacobian: shape (_N_TRI, 2*N+1)
    jac = np.zeros((_N_TRI, 2*N + 1))
    for t in range(_N_TRI):
        i, j, k = _TRI_IDX[t]
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        
        # Gradient w.r.t. point i
        jac[t, 2*i] = 0.5 * (yj - yk)
        jac[t, 2*i+1] = 0.5 * (xj - xk)
        # Gradient w.r.t. point j
        jac[t, 2*j] = 0.5 * (yk - yi)
        jac[t, 2*j+1] = -0.5 * (xk - xi)
        # Gradient w.r.t. point k
        jac[t, 2*k] = 0.5 * (yi - yj)
        jac[t, 2*k+1] = -0.5 * (xi - xj)
        # Gradient w.r.t. alpha
        jac[t, 2*N] = -1.0
    
    return jac

def _solve_nlp_from_init(init_pts, alpha_init, timeout=10.0):
    """Solve the NLP problem from a given initialization using SLSQP.
    Returns (pts, alpha) or None on failure.
    """
    x0 = np.concatenate([init_pts.flatten(), [alpha_init]])
    
    # Bounds: points must be in [0,1] x [0, H], alpha >= 0
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, H))    # y
    bounds.append((0.0, 0.5))  # alpha upper bound (max possible area)
    
    # Constraints: area_t - alpha >= 0 for all t
    # For SLSQP, type 'ineq' means f(x) >= 0
    cons = ({'type': 'ineq', 'fun': _nlp_constraints, 'jac': _nlp_jacobian_constraints})
    
    try:
        res = minimize(
            _nlp_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        if res.success or res.fun < 0:  # If it converged or at least made progress
            pts = res.x[:2*N].reshape(N, 2)
            alpha = res.x[2*N]
            pts = _clamp_all(pts)
            # Verify
            actual_min = _min_area(pts)
            return pts, actual_min
    except Exception:
        pass
    return None

def _solve_nlp_trustcon_from_init(init_pts, alpha_init, timeout=15.0):
    """Solve using trust-constr method as alternative."""
    from scipy.optimize import LinearConstraint, NonlinearConstraint
    
    x0 = np.concatenate([init_pts.flatten(), [alpha_init]])
    
    # Bounds
    lb = np.zeros(2*N + 1)
    ub = np.zeros(2*N + 1)
    for i in range(N):
        lb[2*i] = 0.0
        ub[2*i] = 1.0
        lb[2*i+1] = 0.0
        ub[2*i+1] = H
    lb[2*N] = 0.0
    ub[2*N] = 0.5
    
    # Nonlinear constraint: areas - alpha >= 0
    # For NonlinearConstraint, we need lb_c <= fun(x) <= ub_c
    # We want fun(x) >= 0, so lb_c = 0, ub_c = inf
    nl_con = NonlinearConstraint(
        _nlp_constraints,
        0.0,
        np.inf,
        jac=_nlp_jacobian_constraints
    )
    
    try:
        res = minimize(
            _nlp_objective,
            x0,
            method='trust-constr',
            bounds=list(zip(lb, ub)),
            constraints=[nl_con],
            options={'maxiter': 150, 'gtol': 1e-8, 'xtol': 1e-10, 'disp': False}
        )
        pts = res.x[:2*N].reshape(N, 2)
        alpha = res.x[2*N]
        pts = _clamp_all(pts)
        actual_min = _min_area(pts)
        return pts, actual_min
    except Exception:
        pass
    return None

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _objective_smooth(flat_pts, beta=50.0):
    """Objective to minimize: negative soft-min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def construct():
    start_time = time.time()
    deadline = 138.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_symmetric())
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        init_candidates.append(_initial_placement_random(seed))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: NLP-based optimization (SLSQP) from multiple starts
    for init in init_candidates:
        if time_left() < 12.0:
            break
        init_clamped = _clamp_all(init)
        init_min = _min_area(init_clamped)
        
        # Try SLSQP
        result = _solve_nlp_from_init(init_clamped, init_min * 0.8)
        if result is not None:
            pts, score = result
            if score > best_score:
                best_pts = pts.copy()
                best_score = score
    
    if best_pts is None:
        best_pts = _clamp_all(_initial_placement_symmetric())
        best_score = _min_area(best_pts)
    
    # Phase 2: NLP with trust-constr from the best result
    if time_left() > 20.0:
        result = _solve_nlp_trustcon_from_init(best_pts, best_score * 0.9)
        if result is not None:
            pts, score = result
            if score > best_score:
                best_pts = pts.copy()
                best_score = score
    
    # Phase 3: Multi-start NLP with perturbations
    rng = np.random.default_rng(42)
    n_perturb = 8
    for trial in range(n_perturb):
        if time_left() < 10.0:
            break
        scale = 0.02 + 0.015 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        trial_min = _min_area(trial_pts)
        
        result = _solve_nlp_from_init(trial_pts, trial_min * 0.8)
        if result is not None:
            pts, score = result
            if score > best_score:
                best_pts = pts.copy()
                best_score = score
    
    # Phase 4: Smooth optimization polish (L-BFGS-B)
    if time_left() > 10.0:
        for beta in [40.0, 80.0]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_smooth(x, b),
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
                    best_score = opt_score
            except Exception:
                pass
    
    # Phase 5: Nelder-Mead local search for final refinement
    if time_left() > 8.0:
        try:
            res_nm = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            if nm_score > best_score:
                best_pts = nm_pts.copy()
                best_score = nm_score
        except Exception:
            pass
    
    # Phase 6: Final NLP pass from polished result
    if time_left() > 12.0:
        result = _solve_nlp_from_init(best_pts, best_score * 0.95)
        if result is not None:
            pts, score = result
            if score > best_score:
                best_pts = pts.copy()
                best_score = score
    
    # Phase 7: One more Nelder-Mead pass
    if time_left() > 5.0:
        try:
            res_nm2 = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            nm2_pts = _clamp_all(res_nm2.x.reshape(-1, 2))
            nm2_score = _min_area(nm2_pts)
            if nm2_score > best_score:
                best_pts = nm2_pts.copy()
                best_score = nm2_score
        except Exception:
            pass
    
    # Final clamp and return
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
