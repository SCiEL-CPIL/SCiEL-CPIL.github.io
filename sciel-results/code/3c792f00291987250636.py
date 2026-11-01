import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)
_HULL_CACHE = {}
_HULL_CACHE_MAX = 200

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        key = tuple(np.round(points, 10).flatten())
        if key in _HULL_CACHE:
            return _HULL_CACHE[key]
        ha = float(scipy.spatial.ConvexHull(points).volume)
        if len(_HULL_CACHE) >= _HULL_CACHE_MAX:
            _HULL_CACHE.clear()
        _HULL_CACHE[key] = ha
        return ha
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_ring_config(n, frac=0.5, phase=0.5):
    n_outer = n - 7
    n_inner = 7
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, frac, phase * 2 * np.pi / n_inner)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    """Points on an ellipse."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    # Rotate
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3, np.pi / 7, 2*np.pi/14]:
        starts.append(_regular_polygon(N, 1.0, rot))
    
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.25, 0.5, 0.75]:
            starts.append(_two_ring_config(N, frac, phase))
            
    # Elliptical hulls to explore non-circular convex regions
    for a in [0.9, 1.0, 1.1]:
        for b in [0.7, 0.8, 0.9, 1.0]:
            for rot in [0.0, np.pi/6, np.pi/4, np.pi/3]:
                starts.append(_ellipse_config(N, a, b, rot))
                
    # Three-ring configuration
    n1, n2, n3 = 5, 5, 4
    for r1 in [0.3, 0.4]:
        for r2 in [0.6, 0.7]:
            ring1 = _regular_polygon(n1, 1.0, 0.0)
            ring2 = _regular_polygon(n2, r2, np.pi/n2)
            ring3 = _regular_polygon(n3, r1, np.pi/n3)
            starts.append(np.vstack([ring1, ring2, ring3]))
                
    return starts

def _lse_min_area_and_grad(points, tau=0.01):
    """
    Smooth approximation of min triangle area using Log-Sum-Exp (soft-min)
    with analytical gradient.
    
    soft_min(x) = -tau * log(sum(exp(-x_i / tau)))
    d(soft_min)/dx_i = -exp(-x_i/tau) / sum(exp(-x_j/tau))
    
    Returns (value, gradient w.r.t. points).
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # soft-min value
    min_a = np.min(areas)
    exp_terms = np.exp(-(areas - min_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = -tau * np.log(sum_exp) + min_a
    
    # Weights for gradient: w_i = exp(-a_i/tau) / sum(exp(-a_j/tau))
    weights = exp_terms / sum_exp
    
    # Gradient of cross w.r.t. each point
    # cross = (pj - pi) x (pk - pi)
    # dcross/dpi = -((pk-pi) - (pj-pi)) = -(pk - pj) ... let me recompute
    # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
    # dcross/dpi0 = -(pk1-pi1) + (pj1-pi1) = (pj1-pi1) - (pk1-pi1) = pj1 - pk1
    # dcross/dpi1 = (pk0-pi0) - (pj0-pi0) = pk0 - pj0
    # dcross/dpj0 = (pk1-pi1)
    # dcross/dpj1 = -(pk0-pi0)
    # dcross/dpk0 = -(pj1-pi1)
    # dcross/dpk1 = (pj0-pi0)
    
    # areas = 0.5 * |cross|
    # For simplicity, assume cross > 0 (we can handle sign)
    # d(area)/d(point) = 0.5 * sign(cross) * dcross/d(point)
    
    # Build gradient of areas w.r.t. all points
    n = len(points)
    grad_areas = np.zeros((n, 2))
    
    sign_cross = np.sign(cross)
    half_sign = 0.5 * sign_cross
    
    # For each triangle (i, j, k):
    # darea/dpi = 0.5 * s * [dcross/dpi0, dcross/dpi1]
    # dcross/dpi0 = pj1 - pk1
    # dcross/dpi1 = pk0 - pj0
    # darea/dpj = 0.5 * s * [dcross/dpj0, dcross/dpj1]  
    # dcross/dpj0 = pk1 - pi1
    # dcross/dpj1 = -(pk0 - pi0)
    # darea/dpk = 0.5 * s * [dcross/dpk0, dcross/dpk1]
    # dcross/dpk0 = -(pj1 - pi1)
    # dcross/dpk1 = pj0 - pi0
    
    pi_pts = pi
    pj_pts = pj
    pk_pts = pk
    
    d_pi0 = pj_pts[:, 1] - pk_pts[:, 1]
    d_pi1 = pk_pts[:, 0] - pj_pts[:, 0]
    d_pj0 = pk_pts[:, 1] - pi_pts[:, 1]
    d_pj1 = -(pk_pts[:, 0] - pi_pts[:, 0])
    d_pk0 = -(pj_pts[:, 1] - pi_pts[:, 1])
    d_pk1 = pj_pts[:, 0] - pi_pts[:, 0]
    
    # Weighted sum
    w = half_sign * weights
    
    np.add.at(grad_areas, _I, w[:, np.newaxis] * np.column_stack([d_pi0, d_pi1]))
    np.add.at(grad_areas, _J, w[:, np.newaxis] * np.column_stack([d_pj0, d_pj1]))
    np.add.at(grad_areas, _K, w[:, np.newaxis] * np.column_stack([d_pk0, d_pk1]))
    
    # Gradient of soft_min w.r.t. areas is -weights (from chain rule)
    # Actually: d(soft_min)/d(area_i) = -w_i
    # So grad_soft_min = -sum_i w_i * d(area_i)/d(points)
    # Wait, let me recheck: soft_min = -tau * log(sum(exp(-a/tau)))
    # d(soft_min)/da_i = -tau * (1/sum) * exp(-a_i/tau) * (-1/tau) = exp(-a_i/tau)/sum = w_i
    # So grad_soft_min_wrt_areas = weights (positive)
    # And grad_soft_min_wrt_points = sum_i w_i * d(area_i)/d(points)
    
    grad_points = np.zeros((n, 2))
    np.add.at(grad_points, _I, weights[:, np.newaxis] * np.column_stack([d_pi0, d_pi1]) * 0.5)
    np.add.at(grad_points, _J, weights[:, np.newaxis] * np.column_stack([d_pj0, d_pj1]) * 0.5)
    np.add.at(grad_points, _K, weights[:, np.newaxis] * np.column_stack([d_pk0, d_pk1]) * 0.5)
    
    return lse_val, grad_points

def _smooth_neg_score(flat, tau=0.01):
    """
    Smooth objective for L-BFGS-B.
    Returns negative of (soft_min_triangle_area / Hull_Area).
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    min_a_soft = _lse_min_area_and_grad(pts, tau)[0]
    # We want to maximize min_a / ha
    # So minimize -min_a / ha
    return -min_a_soft / ha

def _smooth_neg_score_and_grad(flat, tau=0.01):
    """
    Smooth objective with gradient for L-BFGS-B.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    
    min_a_soft, grad_min_a = _lse_min_area_and_grad(pts, tau)
    
    # f = -min_a / ha
    # df/dpts = -grad_min_a / ha
    val = -min_a_soft / ha
    grad = -grad_min_a / ha
    return val, grad.flatten()

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score, best_pts = 0.0, _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    # Top candidates for intensive optimization
    # We take top 5 to allow exploration of different local basins
    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau=0.01, use_grad=False):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth and use_grad:
                obj_func = lambda f: _smooth_neg_score_and_grad(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, jac=True, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            elif use_smooth:
                obj_func = lambda f: _smooth_neg_score(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            else:
                obj_func = _neg_score
                opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                        'adaptive': True, 'maxfev': 25000}
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Intensive optimization on top candidates
    # Strategy: 
    # 1. Nelder-Mead on top candidates to find local minima.
    # 2. L-BFGS-B with smooth surrogate (corrected soft-min) on best.
    # 3. Powell on the best found so far.
    # 4. L-BFGS-B with tighter tau for precision.
    # 5. Final tight Nelder-Mead polish.
    
    # Phase 1: Nelder-Mead on top candidates
    for pts in top_candidates:
        if left() <= 12.0:
            break
        try_opt(pts, 'Nelder-Mead', 15000, 1e-13, 1e-10)

    # Phase 2: L-BFGS-B with smooth surrogate + gradient (coarse tau first)
    if left() > 25.0:
        for tau in [0.2, 0.1, 0.05, 0.02]:
            if left() <= 8.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 500, 1e-12, 1e-12, use_smooth=True, tau=tau, use_grad=True)
            # Quick Nelder-Mead polish after each L-BFGS-B step
            if left() > 10.0:
                try_opt(best_pts, 'Nelder-Mead', 3000, 1e-13, 1e-10)

    # Phase 3: Powell on best
    if left() > 15.0:
        try_opt(best_pts, 'Powell', 5000, 1e-13, 1e-11)

    # Phase 4: L-BFGS-B with tight tau for precision (with gradient)
    if left() > 12.0:
        for tau in [0.01, 0.005, 0.002]:
            if left() <= 5.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 400, 1e-14, 1e-14, use_smooth=True, tau=tau, use_grad=True)

    # Phase 5: Final Polish with tight Nelder-Mead
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 10000, 1e-15, 1e-13)
    
    # Phase 6: Ultra-tight final polish
    if left() > 4.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-16, 1e-14)

    return _normalize(best_pts)
