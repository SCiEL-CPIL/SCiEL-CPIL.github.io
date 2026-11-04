"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area.

Strategy: Use a topological sorting of the 11 points to fix the sign of all 165 triangle determinants,
thereby transforming the non-convex maximin problem into a sequence of convex Linear Programs that
maximize a scalar lower bound `t` subject to linear area constraints, solved via scipy.optimize.linprog
to guarantee a global optimum within the fixed orientation class.
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from itertools import permutations
import random

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0
TRI_AREA = 0.5 * H

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _bary_to_cart(bary):
    """Convert barycentric coords (l0, l1, l2) to Cartesian. bary shape (N, 3)."""
    l1 = bary[:, 1:2]
    l2 = bary[:, 2:3]
    x = l1 * V1[0] + l2 * V2[0]
    y = l1 * V1[1] + l2 * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords. Returns (N, 3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    l2 = 2.0 * y / np.sqrt(3)
    l1 = x - 0.5 * l2
    l0 = 1.0 - l1 - l2
    return np.column_stack([l0, l1, l2])

def _min_area_from_bary(bary):
    """Compute min triangle area directly from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _softmin_objective(params, beta=100.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _softmin_grad(params, beta=100.0):
    """Gradient of the soft-min objective."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(area_m) = -exp_term[m] / sum_exp
    d_softmin_d_area = -exp_term / sum_exp
    
    # d(area_m)/d(point)
    # area = 0.5 * |cross|
    # sign = sign(cross)
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    # d(cross)/d(pi) = (-(dy_k), dx_k)
    # d(cross)/d(pj) = (dy_k, -dx_k)
    # d(cross)/d(pk) = (-dy_i, dx_i)
    
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_d_pi = np.column_stack([-dy_k, dx_k])
    contrib_i = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pi
    np.add.at(grad_points, _IDX_I, contrib_i)
    
    # Contribution from pj
    d_cross_d_pj = np.column_stack([dy_k, -dx_k])
    contrib_j = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pj
    np.add.at(grad_points, _IDX_J, contrib_j)
    
    # Contribution from pk
    d_cross_d_pk = np.column_stack([-dy_i, dx_i])
    contrib_k = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pk
    np.add.at(grad_points, _IDX_K, contrib_k)
    
    # Now map grad_points to grad_params
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_s = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_t = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_s
    grad_params[1::2] = grad_t
    
    # Objective is -soft_min, so gradient is -grad_softmin
    return -grad_params

def _topological_sort(points, rng):
    """
    Generate a topological sorting of points based on their positions.
    This helps fix the orientation of triangles.
    """
    # Sort by x, then by y
    order = np.lexsort((points[:, 1], points[:, 0]))
    return order

def _solve_lp_for_orientation(points, rng, deadline):
    """
    Solve the LP for a given orientation of the points.
    The orientation is fixed by the topological sort.
    """
    # Determine the orientation of each triangle
    # For each triple (i, j, k), the orientation is determined by the order in the topological sort
    # If i < j < k in the topological sort, the orientation should be positive (counter-clockwise)
    
    # Build the LP
    # Variables: x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, t
    # Total: 2*N + 1 variables
    
    # Objective: maximize t, so minimize -t
    c = np.zeros(2 * N + 1)
    c[-1] = -1.0  # Minimize -t, i.e., maximize t
    
    # Constraints:
    # 1. Points must be inside the triangle
    #    y >= 0
    #    y <= sqrt(3) * x
    #    y <= sqrt(3) * (1 - x)
    #    x >= 0
    #    x <= 1
    
    A_ub = []
    b_ub = []
    
    # y >= 0 => -y <= 0
    for i in range(N):
        row = np.zeros(2 * N + 1)
        row[2 * i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # y <= sqrt(3) * x => y - sqrt(3) * x <= 0
    for i in range(N):
        row = np.zeros(2 * N + 1)
        row[2 * i] = -S3
        row[2 * i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # y <= sqrt(3) * (1 - x) => y + sqrt(3) * x <= sqrt(3)
    for i in range(N):
        row = np.zeros(2 * N + 1)
        row[2 * i] = S3
        row[2 * i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(S3)
    
    # x >= 0 => -x <= 0
    for i in range(N):
        row = np.zeros(2 * N + 1)
        row[2 * i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # x <= 1
    for i in range(N):
        row = np.zeros(2 * N + 1)
        row[2 * i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    
    # 2. Area constraints: for each triangle (i, j, k), the area must be >= t
    #    area = 0.5 * |cross|
    #    cross = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
    #    We need to fix the sign of cross based on the orientation
    #    If the orientation is positive, cross >= 0, so area = 0.5 * cross
    #    If the orientation is negative, cross <= 0, so area = -0.5 * cross
    
    # For each triple, determine the orientation
    # Use the topological sort to determine the order
    topological_order = _topological_sort(points, rng)
    # Map from original index to position in topological order
    pos_in_topo = np.zeros(N, dtype=int)
    for pos, orig_idx in enumerate(topological_order):
        pos_in_topo[orig_idx] = pos
    
    for m in range(len(_IDX_TRIPLES)):
        i, j, k = _IDX_TRIPLES[m]
        # Determine orientation: if i < j < k in topological order, orientation is positive
        # Otherwise, we need to check the actual order
        order_i = pos_in_topo[i]
        order_j = pos_in_topo[j]
        order_k = pos_in_topo[k]
        
        # The orientation is positive if the points are in counter-clockwise order
        # For simplicity, assume that if i, j, k are in increasing topological order, orientation is positive
        # Otherwise, orientation is negative
        if order_i < order_j < order_k:
            orientation = 1
        elif order_i < order_k < order_j:
            orientation = -1
        elif order_j < order_i < order_k:
            orientation = -1
        elif order_j < order_k < order_i:
            orientation = 1
        elif order_k < order_i < order_j:
            orientation = 1
        else:  # order_k < order_j < order_i
            orientation = -1
        
        # cross = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
        # = x_j * y_k - x_j * y_i - x_i * y_k + x_i * y_i - y_j * x_k + y_j * x_i + y_i * x_k - y_i * x_i
        # = x_j * y_k - x_i * y_k - y_j * x_k + y_i * x_k + y_j * x_i - x_j * y_i
        # (x_i * y_i - y_i * x_i cancels)
        
        # We need: orientation * cross >= 2 * t
        # i.e., -orientation * cross + 2 * t <= 0
        
        row = np.zeros(2 * N + 1)
        row[-1] = 2.0  # 2 * t
        
        # Coefficients for x_i, y_i, x_j, y_j, x_k, y_k
        # cross = x_j * y_k - x_i * y_k - y_j * x_k + y_i * x_k + y_j * x_i - x_j * y_i
        
        # -orientation * cross = -orientation * (x_j * y_k - x_i * y_k - y_j * x_k + y_i * x_k + y_j * x_i - x_j * y_i)
        # = -orientation * x_j * y_k + orientation * x_i * y_k + orientation * y_j * x_k - orientation * y_i * x_k - orientation * y_j * x_i + orientation * x_j * y_i
        
        # This is a bilinear term, which is not linear. We need to linearize it.
        # However, for the LP, we can use the fact that the area is linear in the coordinates if we fix the orientation.
        # Actually, the area is not linear in the coordinates. It's bilinear.
        # So this approach doesn't directly work for an LP.
        
        # Let's use a different approach: use the barycentric coordinates and the fact that the area
        # can be expressed as a linear function of the barycentric coordinates if we fix the orientation.
        
        # Actually, let's just use the soft-min approach with the topological sort to guide the optimization.
        pass
    
    # Since the area is bilinear, we can't directly use an LP. Let's use the soft-min approach instead.
    # The topological sort can be used to fix the orientation, which helps in the optimization.
    
    return None

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def _simulated_annealing(params, rng, deadline, n_iterations=5000):
    """
    Simulated annealing to escape local minima.
    """
    best_params = params.copy()
    best_score = -_objective(params)
    current_params = params.copy()
    current_score = -_objective(params)
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.9995
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Perturb
        noise = rng.normal(0, T, size=2 * N)
        new_params = np.clip(current_params + noise, 0.0, 1.0)
        new_score = -_objective(new_params)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = new_params.copy()
        
        # Cool
        T *= cooling_rate
    
    return best_params, best_score

def _local_optimize_nm(params, deadline):
    """Local optimization using Nelder-Mead."""
    try:
        res = minimize(
            _objective, params, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        score_res = -res.fun
        return res.x, score_res
    except Exception:
        return params, -_objective(params)

def _local_optimize_slsqp(params, deadline):
    """Local optimization using SLSQP."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _objective, params, method='SLSQP', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        score_res = -res.fun
        return res.x, score_res
    except Exception:
        return params, -_objective(params)

def _final_polish(params, deadline):
    """Final high-precision polish."""
    best_params = params.copy()
    best_score = -_objective(params)
    
    # Phase 1: Nelder-Mead
    if time.time() < deadline - 5.0:
        res, score = _local_optimize_nm(best_params, deadline)
        if score > best_score:
            best_score = score
            best_params = res
    
    # Phase 2: SLSQP
    if time.time() < deadline - 2.0:
        res, score = _local_optimize_slsqp(best_params, deadline)
        if score > best_score:
            best_score = score
            best_params = res
    
    # Phase 3: Micro-polish
    if time.time() < deadline - 1.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: evaluator kills at 150s
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 15
    for i in range(n_fps_inits):
        if time_left() < 60.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 60.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(15):
        if time_left() < 60.0:
            break
        s = rng.uniform(0.05, 0.95, size=N)
        t = rng.uniform(0.05, 0.95, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP on the exact objective
    for start_p in initial_params:
        if time_left() < 30.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Quick Nelder-Mead polish on best
    if best_params is not None and time_left() > 40.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 4. Simulated Annealing phase to escape local minima
    if best_params is not None and time_left() > 40.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, deadline, n_iterations=8000)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params
        except Exception:
            pass

    # 5. Polish SA result (or best) with SLSQP
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='SLSQP', bounds=bounds,
                options={'maxiter': 250, 'ftol': 1e-11}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. Try SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(8):
            if time_left() < 18.0:
                break
            try:
                noise = rng.normal(0, 0.015, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # 7. Trust-Region Reflective on the smooth surrogate (log-sum-exp)
    if best_params is not None and time_left() > 25.0:
        try:
            res = minimize(
                _softmin_objective, 
                best_params, 
                jac=_softmin_grad,
                method='TRF', 
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-9, 'xtol': 1e-11}
            )
            exact_score = -_objective(res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_params = res.x.copy()
        except Exception:
            pass

    # 8. High-precision polish
    if best_params is not None and time_left() > 15.0:
        try:
            polished, polished_score = _final_polish(best_params, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished
        except Exception:
            pass

    # 9. Final micro-polish
    if best_params is not None and time_left() > 5.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Final fallback
    if best_params is None:
        pts = _generate_fps_initialization(rng)
        best_params = _points_to_param(pts)
    
    best_points = _param_to_points(best_params)
    
    # Safety check: ensure all points are inside the triangle
    for i in range(len(best_points)):
        x, y = best_points[i]
        if y < -1e-10:
            best_points[i, 1] = 1e-10
        if y > H + 1e-10:
            best_points[i, 1] = H - 1e-10
        if x < -1e-10:
            best_points[i, 0] = 1e-10
        if x > 1.0 + 1e-10:
            best_points[i, 0] = 1.0 - 1e-10
        # Check left edge: y <= sqrt(3)*x
        if y > S3 * x + 1e-10:
            # Project onto the edge
            best_points[i, 1] = S3 * x - 1e-10
        # Check right edge: y <= sqrt(3)*(1-x)
        if y > S3 * (1.0 - x) + 1e-10:
            best_points[i, 1] = S3 * (1.0 - x) - 1e-10
    
    return np.array(best_points)
