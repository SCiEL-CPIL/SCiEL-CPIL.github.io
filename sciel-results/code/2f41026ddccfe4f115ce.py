import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

# Precompute row/col indices for gradient computation
_I_IDX = _IDX_TRIPLES[:, 0]
_J_IDX = _IDX_TRIPLES[:, 1]
_K_IDX = _IDX_TRIPLES[:, 2]

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    Pi = P[_I_IDX]
    Pj = P[_J_IDX]
    Pk = P[_K_IDX]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _det_grad(bary):
    """
    Compute gradient of all determinants w.r.t. barycentric coords (u, v).
    Returns array of shape (n_triples, 2*N).
    
    For triple (i,j,k):
    det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    
    Gradient w.r.t. u_i: v_j*w_k - w_j*v_k - (u_j*w_k - w_j*u_k) - (u_j*v_k - v_j*u_k)
    Gradient w.r.t. v_i: -(u_j*w_k - w_j*u_k) - (u_j*v_k - v_j*u_k)
    By symmetry for j and k.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I_IDX]; vi = v[_I_IDX]; wi = w[_I_IDX]
    uj = u[_J_IDX]; vj = v[_J_IDX]; wj = w[_J_IDX]
    uk = u[_K_IDX]; vk = v[_K_IDX]; wk = w[_K_IDX]
    
    # Compute the three "cross" terms for each triple
    # A = vj*wk - wj*vk (coeff of ui in det)
    # B = uj*wk - wj*uk (coeff of -vi in det, with sign)
    # C = uj*vk - vj*uk (coeff of wi in det)
    A = vj * wk - wj * vk
    B = uj * wk - wj * uk
    C = uj * vk - vj * uk
    
    # Gradient w.r.t. u_i: A - B - C
    # Gradient w.r.t. v_i: -B - C  ... wait let me recompute
    # det = ui*A - vi*B + wi*C
    # d(det)/du_i = A - B - C  (since d(ui)/du_i=1, d(wi)/du_i=-1)
    # d(det)/dv_i = -B - C     (since d(vi)/dv_i=1, d(wi)/dv_i=-1)
    # By cyclic symmetry for j and k
    
    grad = np.zeros((n_triples, 2 * N))
    
    # For point i:
    grad[:, 2*_I_IDX] = A - B - C
    grad[:, 2*_I_IDX + 1] = -B - C
    
    # For point j:
    # d(det)/du_j = -B + A + C  (d(wj)/du_j = -1, d(vj)/du_j = 0, d(uj)/du_j = 1)
    # Actually let me be more careful.
    # det = ui*(vj*wk - wj*vk) - vi*(uj*wk - wj*uk) + wi*(uj*vk - vj*uk)
    # d/d(uj): ui*(0*wk - wj*0) - vi*(1*wk - wj*0) + wi*(1*vk - vj*0) + d(wi)/duj * (uj*vk - vj*uk)
    #        = -vi*wk + wi*vk + (-1)*(uj*vk - vj*uk)
    #        = -vi*wk + wi*vk - uj*vk + vj*uk
    # Let me just use: d(det)/du_j = -B + A + C where we swap indices appropriately
    # Actually by symmetry: d(det)/du_j = -B_j + A_j + C_j where we relabel
    # Let me compute directly:
    # d(det)/du_j: 
    #   from ui*(vj*wk - wj*vk): d(wj)/duj = -1, so ui*(vj*0 - (-1)*vk) = ui*vk
    #   from -vi*(uj*wk - wj*uk): d(uj)/duj = 1, so -vi*(1*wk - 0*uk) = -vi*wk
    #   from wi*(uj*vk - vj*uk): d(uj)/duj = 1, d(wi)/duj = -1
    #     = 1*(1*vk - vj*0) + (-1)*(uj*vk - vj*uk) = vk - uj*vk + vj*uk
    # Total: ui*vk - vi*wk + vk - uj*vk + vj*uk
    # Hmm, this is getting complex. Let me just compute numerically or use a simpler approach.
    
    # Actually, the cleanest way: the determinant is multilinear. 
    # d(det)/du_j = det with uj=1, others 0 for the j-slot contribution... 
    # No, let me just do it properly with the formula.
    
    # For a triple (i,j,k), writing det in expanded form:
    # det = ui*vj*wk + vi*wj*uk + wi*uj*vk - wi*vj*uk - ui*wj*vk - vi*uj*wk
    
    # d/d(ui) = vj*wk - wj*vk
    # d/d(vi) = wj*uk - uj*wk  
    # d/d(wi) = uj*vk - vj*uk
    
    # But w = 1-u-v, so:
    # d/d(u_m) for m=i: vj*wk - wj*vk + (uj*vk - vj*uk)  [because dwi/dui = -1]
    #                  = A + C  ... wait no
    # d(det)/du_i = d(det)/du_i|direct + d(det)/dwi * d(wi)/du_i
    #             = (vj*wk - wj*vk) + (uj*vk - vj*uk)*(-1)
    #             = A - C
    # d(det)/dv_i = (wj*uk - uj*wk) + (uj*vk - vj*uk)*(-1)
    #             = -B - C  ... where B = uj*wk - wj*uk, so wj*uk - uj*wk = -B
    #             = -B - C
    
    # For j:
    # d(det)/du_j = d/d(vj)*d(vj)/du_j + d/d(wj)*d(wj)/du_j + d/d(uj)*d(uj)/du_j
    # vj doesn't depend on uj, wj = 1-uj-vj so d(wj)/duj = -1, uj depends on uj
    # = 0 + (uj*vk - vj*uk)*(-1) + (vj*wk - wj*vk)*1
    # Wait, I need to be more careful. Let me use the expanded form:
    # det = ui*vj*wk + vi*wj*uk + wi*uj*vk - wi*vj*uk - ui*wj*vk - vi*uj*wk
    # d/d(uj): 
    #   ui*vj*wk: no uj -> 0
    #   vi*wj*uk: d(wj)/duj = -1 -> -vi*uk
    #   wi*uj*vk: d(uj)/duj = 1, d(wi)/duj = -1 -> wi*vk + (-1)*uj*vk = wi*vk - uj*vk
    #   -wi*vj*uk: d(wi)/duj = -1 -> -(-1)*vj*uk = vj*uk
    #   -ui*wj*vk: d(wj)/duj = -1 -> -ui*(-1)*vk = ui*vk
    #   -vi*uj*wk: d(uj)/duj = 1 -> -vi*wk
    # Total: -vi*uk + wi*vk - uj*vk + vj*uk + ui*vk - vi*wk
    
    # This is getting messy. Let me just compute the gradient numerically for correctness
    # and use it for L-BFGS-B.
    
    # Actually, for L-BFGS-B, we can just use the finite-difference gradient or 
    # provide the analytic gradient. Let me provide a correct analytic gradient.
    
    # Using the expanded form and chain rule properly:
    # Let me define for each triple the partial derivatives.
    
    # d(det)/du_i = vj*wk - wj*vk - (uj*vk - vj*uk) = A - C
    # d(det)/dv_i = wj*uk - uj*wk - (uj*vk - vj*uk) = -B - C
    
    # d(det)/du_j = -vi*uk + wi*vk - uj*vk + vj*uk + ui*vk - vi*wk
    # Let me simplify: = ui*vk - vi*wk - vi*uk + vj*uk + wi*vk - uj*vk
    #                   = vk*(ui + wi - uj) + uk*(-vi + vj) - vi*wk
    # Hmm, let me just verify with a simple case.
    
    # I'll use a different approach: compute gradient by treating w as independent
    # then applying the constraint w = 1-u-v.
    
    # Partial derivatives treating (u,v,w) as independent:
    # d/d(ui) = vj*wk - wj*vk
    # d/d(vi) = wj*uk - uj*wk
    # d/d(wi) = uj*vk - vj*uk
    # Similarly for j and k.
    
    # Now, since w_m = 1 - u_m - v_m:
    # d/d(u_m) = d/d(u_m)| + d/d(w_m)| * d(w_m)/d(u_m) = d/d(u_m)| - d/d(w_m)|
    # d/d(v_m) = d/d(v_m)| + d/d(w_m)| * d(w_m)/d(v_m) = d/d(v_m)| - d/d(w_m)|
    
    # So for each point m in {i,j,k}:
    # d(det)/du_m = [d(det)/du_m treating u,v,w independent] - [d(det)/dw_m treating u,v,w independent]
    # d(det)/dv_m = [d(det)/dv_m treating u,v,w independent] - [d(det)/dw_m treating u,v,w independent]
    
    # For point i:
    # d/d(ui)| = vj*wk - wj*vk = A
    # d/d(wi)| = uj*vk - vj*uk = C
    # d/d(vi)| = wj*uk - uj*wk = -B
    # So d(det)/du_i = A - C
    #    d(det)/dv_i = -B - C
    
    # For point j:
    # d/d(uj)| = wj*vk - wj*... wait, by symmetry:
    # d/d(uj)| = w_i * v_k - v_i * w_k  ... no
    # Let me just use: for point j, the roles are shifted.
    # d/d(uj)| (treating u,v,w independent) = w_i * v_k - v_i * w_k
    # d/d(wj)| = u_i * v_k - v_i * u_k
    # d/d(vj)| = u_i * w_k - w_i * u_k
    
    # So d(det)/du_j = (w_i*vk - vi*wk) - (ui*vk - vi*uk)
    #                = w_i*vk - vi*wk - ui*vk + vi*uk
    # d(det)/dv_j = (ui*wk - wi*uk) - (ui*vk - vi*uk)
    #              = ui*wk - wi*uk - ui*vk + vi*uk
    
    # For point k:
    # d/d(uk)| = v_i * w_j - w_i * v_j
    # d/d(wk)| = u_i * v_j - v_i * u_j
    # d/d(vk)| = u_i * w_j - w_i * u_j
    # d(det)/du_k = (vi*wj - wi*vj) - (ui*vj - vi*uj)
    # d(det)/dv_k = (ui*wj - wi*uj) - (ui*vj - vi*uj)
    
    # Let me just compute all of these explicitly:
    
    # For i:
    gui_u = A - C  # vj*wk - wj*vk - (uj*vk - vj*uk)
    gui_v = -B - C  # (wj*uk - uj*wk) - (uj*vk - vj*uk)
    
    # For j:
    # d/d(uj)| = wi*vk - vi*wk
    # d/d(wj)| = ui*vk - vi*uk
    # d/d(vj)| = ui*wk - wi*uk
    guj_u = (wi * vk - vi * wk) - (ui * vk - vi * uk)
    guj_v = (ui * wk - wi * uk) - (ui * vk - vi * uk)
    
    # For k:
    # d/d(uk)| = vi*wj - wi*vj
    # d/d(wk)| = ui*vj - vi*uj
    # d/d(vk)| = ui*wj - wi*uj
    guk_u = (vi * wj - wi * vj) - (ui * vj - vi * uj)
    guk_v = (ui * wj - wi * uj) - (ui * vj - vi * uj)
    
    grad[:, 2*_I_IDX] += gui_u
    grad[:, 2*_I_IDX + 1] += gui_v
    grad[:, 2*_J_IDX] += guj_u
    grad[:, 2*_J_IDX + 1] += guj_v
    grad[:, 2*_K_IDX] += guk_u
    grad[:, 2*_K_IDX + 1] += guk_v
    
    return grad

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _score_bary(bary):
    """Compute the score for a barycentric configuration."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return min_det / (2 * TRI_AREA)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _clamp_bary(bary):
    """Clamp barycentric coords to valid simplex."""
    u = np.clip(bary[:, 0], 0.0, 1.0)
    v = np.clip(bary[:, 1], 0.0, 1.0)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _softmin_objective(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    Returns scalar objective value.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _softmin_grad(x, beta, penalty_lambda=100.0):
    """
    Gradient of the soft-min objective.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty gradient
    grad_pen = np.zeros_like(x)
    excess = np.maximum(0, u + v - 1.0)
    if np.any(excess > 0):
        grad_pen[0::2] += 2 * penalty_lambda * excess
        grad_pen[1::2] += 2 * penalty_lambda * excess
    excess_u = np.maximum(0, -u)
    if np.any(excess_u > 0):
        grad_pen[0::2] += 2 * penalty_lambda * excess_u
    excess_v = np.maximum(0, -v)
    if np.any(excess_v > 0):
        grad_pen[1::2] += 2 * penalty_lambda * excess_v
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    m = np.min(areas)
    shifted = areas - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights, sum to 1
    
    # Gradient of smooth_min = m - (1/beta) * log(sum_exp)
    # d(smooth_min)/dx = d(m)/dx - (1/beta) * d(log(sum_exp))/dx
    # d(log(sum_exp))/dx = sum_k weights_k * d(area_k)/dx
    # d(m)/dx: m is the min, so gradient is the gradient of the argmin area
    
    # For the min term, use the gradient of the minimum area
    min_idx = np.argmin(areas)
    grad_m = np.zeros_like(x)
    
    # Get gradient of the minimum determinant
    det_grads = _det_grad(bary)  # shape (n_triples, 2*N)
    min_det_grad = det_grads[min_idx]
    
    # The sign of the determinant matters for the absolute value
    sign_min = np.sign(dets[min_idx])
    if sign_min == 0:
        sign_min = 1.0
    grad_m = sign_min * min_det_grad
    
    # Gradient of the log-sum-exp term
    grad_lse = np.zeros_like(x)
    for k in range(n_triples):
        if weights[k] > 1e-15:
            sign_k = np.sign(dets[k])
            if sign_k == 0:
                sign_k = 1.0
            grad_lse += weights[k] * sign_k * det_grads[k]
    
    grad_smooth_min = grad_m - (1.0 / beta) * grad_lse
    
    return -grad_smooth_min + grad_pen

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _objective_hard(x):
    """
    Hard objective: minimize -min_area.
    x: (2N,) array.
    """
    bary = _clamp_bary(x.reshape(N, 2))
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return -min_area

def _local_optimize_lbfgs_softmin(x0, beta=50.0, maxiter=300, gtol=1e-10):
    """Run L-BFGS-B with soft-min objective and gradient from x0."""
    try:
        res = minimize(
            _softmin_objective, x0, method='L-BFGS-B',
            jac=_softmin_grad,
            args=(beta,),
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': 5000}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_nm(x0, maxiter=500, xatol=1e-10, fatol=1e-10):
    """Run Nelder-Mead local optimization from x0 with hard objective."""
    try:
        res = minimize(
            _objective_hard, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=300):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_hard, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxfun': 5000}
        )
        return res.x
    except Exception:
        return x0

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    # Optimize each initial point with a chain of methods
    for init in inits:
        if time_left() < 10.0:
            break
        
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        # Phase 1: L-BFGS-B with soft-min (smooth, fast)
        if time_left() > 15.0:
            for beta in [30.0, 80.0, 200.0]:
                if time_left() < 10.0:
                    break
                x_opt = _local_optimize_lbfgs_softmin(x0, beta=beta, maxiter=200, gtol=1e-10)
                x_opt = _clamp_bary(x_opt.reshape(N, 2)).flatten()
                s = _score_bary(x_opt.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
                x0 = x_opt
        
        # Phase 2: Nelder-Mead with hard objective (nonsmooth but can escape)
        if time_left() > 10.0:
            x_opt = _local_optimize_nm(x0, maxiter=500)
            x_opt = _clamp_bary(x_opt.reshape(N, 2)).flatten()
            s = _score_bary(x_opt.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = x_opt.copy()
            x0 = x_opt
        
        # Phase 3: SLSQP polish
        if time_left() > 8.0:
            x_opt = _local_optimize_slsqp(x0, maxiter=300)
            x_opt = _clamp_bary(x_opt.reshape(N, 2)).flatten()
            s = _score_bary(x_opt.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = x_opt.copy()
            x0 = x_opt
    
    # Final high-precision polish on the best solution
    if best_x is not None and time_left() > 15.0:
        # Multiple soft-min passes with increasing beta
        for beta in [100.0, 300.0, 1000.0]:
            if time_left() < 10.0:
                break
            x_opt = _local_optimize_lbfgs_softmin(best_x, beta=beta, maxiter=500, gtol=1e-12)
            x_opt = _clamp_bary(x_opt.reshape(N, 2)).flatten()
            s = _score_bary(x_opt.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = x_opt.copy()
        
        # Final NM polish
        if time_left() > 8.0:
            x_opt = _local_optimize_nm(best_x, maxiter=1000, xatol=1e-12, fatol=1e-12)
            x_opt = _clamp_bary(x_opt.reshape(N, 2)).flatten()
            s = _score_bary(x_opt.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = x_opt.copy()
    
    # If we found a good solution, convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)
