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
    """Fully vectorized computation of all triangle areas."""
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

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_area_safe(soft_min_val, m)

def _soft_min_area_safe(val, m):
    """Ensure soft min doesn't exceed actual min."""
    return min(val, m)

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    pts_clamped = _clamp_all(pts)
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    return -area_val + 100.0 * penalty

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    """
    alpha = x[0]
    return -alpha

def _nlp_constraints(x):
    """
    Constraints for NLP: each triangle area >= alpha
    Returns array of (area_i - alpha) for all i
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    areas = _all_areas(_clamp_all(pts))
    return areas - alpha

def _nlp_bounds():
    """
    Bounds for NLP: alpha in [0, 0.5], points in triangle
    """
    bounds = [(0.0, 0.5)]  # alpha
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, H))    # y
    return bounds

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for differential evolution
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # Phase 1: Run Differential Evolution for global exploration
    best_pts = None
    try:
        if time_left() > 30.0:
            res_de = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=12,
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
    
    # Phase 2: NLP refinement using SLSQP with constraints
    # This directly maximizes the minimum area via alpha
    if time_left() > 15.0:
        try:
            # Initialize with current best points and estimated alpha
            alpha_init = current_best * 0.9
            x_init = np.concatenate([[alpha_init], best_pts.flatten()])
            
            res_nlp = minimize(
                _nlp_objective,
                x_init,
                method='SLSQP',
                bounds=_nlp_bounds(),
                constraints=[{'type': 'ineq', 'fun': _nlp_constraints}],
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if res_nlp.success or res_nlp.fun < 0:
                alpha_opt = res_nlp.x[0]
                pts_nlp = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                min_area_nlp = _min_area(pts_nlp)
                if min_area_nlp > current_best:
                    best_pts = pts_nlp
                    current_best = min_area_nlp
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B on soft objective for smooth refinement
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=50.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > current_best:
                best_pts = polished_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead on hard objective for final refinement
    if time_left() > 8.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 5: Another NLP pass if time permits
    if time_left() > 8.0:
        try:
            alpha_init = current_best * 0.95
            x_init = np.concatenate([[alpha_init], best_pts.flatten()])
            
            res_nlp2 = minimize(
                _nlp_objective,
                x_init,
                method='SLSQP',
                bounds=_nlp_bounds(),
                constraints=[{'type': 'ineq', 'fun': _nlp_constraints}],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_nlp2.success or res_nlp2.fun < 0:
                pts_nlp2 = _clamp_all(res_nlp2.x[1:].reshape(-1, 2))
                min_area_nlp2 = _min_area(pts_nlp2)
                if min_area_nlp2 > current_best:
                    best_pts = pts_nlp2
                    current_best = min_area_nlp2
        except Exception:
            pass
    
    # Phase 6: Final Nelder-Mead polish
    if time_left() > 3.0:
        try:
            res_nm2 = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm2.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass
    
    return _clamp_all(best_pts)

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

# EVOLVE-BLOCK-END
