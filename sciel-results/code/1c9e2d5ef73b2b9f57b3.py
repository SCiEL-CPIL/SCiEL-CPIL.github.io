import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0  # wall-clock guard (evaluator kills at 150s)

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
_NTRI = _TRI.shape[0]

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
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _hull_area_grad(points):
    """
    Returns the gradient of the convex hull area with respect to the points.
    Assumes points are in general position. If a point is inside the hull, 
    its gradient is 0. If on the hull, it is computed based on neighbors.
    """
    grad = np.zeros((len(points), 2))
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_verts = hull.vertices
        # Sort hull vertices in order
        # scipy.spatial.ConvexHull.vertices are in counterclockwise order
        # but we need to ensure they are connected in a loop
        # The .simplices attribute gives the facets (edges in 2D).
        # In 2D, the hull boundary is a sequence of edges.
        # We can reconstruct the order from the simplices.
        
        # Simpler approach: 
        # The gradient of the area of a convex polygon with vertices v_0, ..., v_{k-1}
        # with respect to v_i is 0.5 * (v_{i-1} - v_{i+1}) rotated?
        # Area = 0.5 * sum(v_i x v_{i+1})
        # dA/dv_i = 0.5 * (v_{i-1} x e_x? No.)
        # Let's use the formula: dA/dx_i = 0.5 * (y_{i-1} - y_{i+1})
        # dA/dy_i = 0.5 * (x_{i+1} - x_{i-1})
        
        # We need the indices of the neighbors of each hull vertex in the hull polygon.
        # hull.simplices in 2D is an array of shape (k, 2).
        # Each simplex is an edge (i, j).
        # We can build a graph of the hull.
        
        edges = hull.simplices
        k = len(hull_verts)
        
        # Map vertex index to its position in the hull order?
        # Actually, we just need the neighbors.
        # For each vertex in hull_verts, find its two neighbors in the hull.
        
        # Build adjacency list for the hull graph
        adj = {v: [] for v in hull_verts}
        for edge in edges:
            u, v = edge
            adj[u].append(v)
            adj[v].append(u)
            
        # For each hull vertex, compute gradient
        for v in hull_verts:
            neighbors = adj[v]
            if len(neighbors) != 2:
                continue # Should not happen for a proper hull
            v_prev = neighbors[0]
            v_next = neighbors[1]
            
            p_prev = points[v_prev]
            p_next = points[v_next]
            p_curr = points[v]
            
            # dA/dx_v = 0.5 * (y_prev - y_next)
            # dA/dy_v = 0.5 * (x_next - x_prev)
            grad[v, 0] = 0.5 * (p_prev[1] - p_next[1])
            grad[v, 1] = 0.5 * (p_next[0] - p_prev[0])
            
    except Exception:
        pass
    return grad

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

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    """Score for two-nested-heptagons parameterization."""
    try:
        r_inner, phi = params
        if r_inner < 0:
            r_inner = 0.0
        points = _two_heptagons(r_inner, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_heptagons(params):
    return -_score_two_heptagons(params)

def _exact_maxmin_obj_grad(points):
    """
    Returns (value, grad) for f(points) = min_triangle_area(points) / hull_area(points).
    We minimize -f(points).
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    # Gradient of cross product w.r.t points
    # C = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # dC/dxi = -(yk-yi) + (yj-yi) = yj - yk
    # dC/dyi = (xj-xi) - (xk-xi) = xj - xk
    # dC/dxj = (yk-yi)
    # dC/dyj = -(xk-xi)
    # dC/dxk = -(yj-yi)
    # dC/dyk = (xj-xi)
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v1[:, 0] - v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    wh = 0.5 * w * signs

    grad_min_area = np.zeros((N, 2))
    np.add.at(grad_min_area, (_I, 0), wh * g_a_x)
    np.add.at(grad_min_area, (_I, 1), wh * g_a_y)
    np.add.at(grad_min_area, (_J, 0), wh * g_b_x)
    np.add.at(grad_min_area, (_J, 1), wh * g_b_y)
    np.add.at(grad_min_area, (_K, 0), wh * g_c_x)
    np.add.at(grad_min_area, (_K, 1), wh * g_c_y)

    # Gradient of hull area
    grad_hull = _hull_area_grad(points)

    # f = m / ha
    # grad f = (grad_m * ha - m * grad_ha) / ha^2
    # We want to minimize -f, so grad (-f) = - (grad_m * ha - m * grad_ha) / ha^2
    #                 = (m * grad_ha - grad_m * ha) / ha^2
    
    val = m / ha
    grad_val = (m * grad_hull - grad_min_area * ha) / (ha * ha)
    
    return -val, -grad_val

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for soft-min objective."""
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros((N, 2))
    except Exception:
        return 0.0, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t. areas
    g_areas = -probs
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v1[:, 0] - v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    coeff = g_areas * 0.5 * signs
    
    grad_soft_min = np.zeros((N, 2))
    np.add.at(grad_soft_min, (_I, 0), coeff * g_a_x)
    np.add.at(grad_soft_min, (_I, 1), coeff * g_a_y)
    np.add.at(grad_soft_min, (_J, 0), coeff * g_b_x)
    np.add.at(grad_soft_min, (_J, 1), coeff * g_b_y)
    np.add.at(grad_soft_min, (_K, 0), coeff * g_c_x)
    np.add.at(grad_soft_min, (_K, 1), coeff * g_c_y)
    
    grad_hull = _hull_area_grad(points)
    
    val = -soft_min_val / ha
    # f = soft_min / ha
    # grad f = (grad_soft_min * ha - soft_min * grad_hull) / ha^2
    # grad (-f) = - (grad_soft_min * ha - soft_min * grad_hull) / ha^2
    grad_val = (soft_min_val * grad_hull - grad_soft_min * ha) / (ha * ha)
    
    return val, -grad_val

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    def try_update(points):
        nonlocal best_score, best_pts
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_pts = points.copy()
        return sc
    
    # ---- Phase 0: Two-nested-heptagons 2D global search ----
    if left() > 5.0:
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        if left() > 5.0:
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            gr = 0.6180339887498949
            a_r, b_r = max(0.0, best_r - 0.1), min(1.0, best_r + 0.1)
            c_r = b_r - gr * (b_r - a_r)
            d_r = a_r + gr * (b_r - a_r)
            fc = _score_two_heptagons([c_r, best_phi])
            fd = _score_two_heptagons([d_r, best_phi])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_r = c_r
                    c_r = d_r
                    fc = fd
                    d_r = b_r - gr * (b_r - a_r)
                    fd = _score_two_heptagons([d_r, best_phi])
                else:
                    b_r = d_r
                    d_r = c_r
                    fd = fc
                    c_r = a_r + gr * (b_r - a_r)
                    fc = _score_two_heptagons([c_r, best_phi])
            best_r = (a_r + b_r) / 2
            
            a_p, b_p = 0.0, np.pi / 7
            c_p = b_p - gr * (b_p - a_p)
            d_p = a_p + gr * (b_p - a_p)
            fc = _score_two_heptagons([best_r, c_p])
            fd = _score_two_heptagons([best_r, d_p])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_p = c_p
                    c_p = d_p
                    fc = fd
                    d_p = b_p - gr * (b_p - a_p)
                    fd = _score_two_heptagons([best_r, d_p])
                else:
                    b_p = d_p
                    d_p = c_p
                    fd = fc
                    c_p = a_p + gr * (b_p - a_p)
                    fc = _score_two_heptagons([best_r, c_p])
            best_phi = (a_p + b_p) / 2
            
            pts = _two_heptagons(best_r, best_phi)
            try_update(pts)
    
    # ---- Phase 1: Gradient-based refinement from best structured start ----
    if left() > 30.0:
        starts = [best_pts.copy()]
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
        ))))
        
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6 * np.cos(a2), 0.6 * np.sin(a2))),
            np.column_stack((0.25 * np.cos(a3), 0.25 * np.sin(a3)))
        ))))
        
        rng2 = np.random.default_rng(123)
        for _ in range(3):
            if left() <= 10.0:
                break
            jittered = best_pts + rng2.normal(0, 0.08, (N, 2))
            starts.append(_normalize(jittered))
        
        def try_opt(x0, method, maxiter, fatol, xatol, use_grad=True):
            nonlocal best_score, best_pts
            if left() <= 3.0:
                return
            try:
                if use_grad:
                    def obj_and_grad(flat):
                        val, grad = _exact_maxmin_obj_grad(flat.reshape(-1, 2))
                        return val, grad.flatten()
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                            'maxfev': 8000, 'maxls': 30}
                    res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                                  jac=True, options=opts)
                else:
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 12000}
                    if method == 'Nelder-Mead':
                        opts['adaptive'] = True
                    res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
        
        for start in starts:
            if left() <= 8.0:
                break
            try_opt(start, 'L-BFGS-B', 1500, 1e-12, 1e-11, use_grad=True)
            if left() <= 6.0:
                break
            try_opt(start, 'Nelder-Mead', 2000, 1e-12, 1e-9, use_grad=False)
            
    # ---- Phase 2: Soft-min annealing with correct gradient ----
    if left() > 15.0:
        tau_vals = [0.05, 0.02, 0.01, 0.005, 0.002]
        for tau in tau_vals:
            if left() <= 4.0:
                break
            try:
                def obj_and_grad(flat):
                    val, grad = _soft_min_score_grad(flat, tau)
                    return val, grad.flatten()
                res = scipy.optimize.minimize(obj_and_grad, best_pts.flatten(), method='L-BFGS-B',
                                              jac=True, options={'maxiter': 800, 'fatol': 1e-12, 'xatol': 1e-11})
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
                
    # ---- Phase 3: Final polish ----
    if left() > 5.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-13, use_grad=True)
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-15, 1e-14, use_grad=False)

    return _normalize(best_pts)
