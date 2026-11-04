"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2.0)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2.0)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    """Project each row (b, c) into the barycentric simplex {b>=0, c>=0, b+c<=1}."""
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, S3 / 2.0)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    """Twice-area determinants for all 165 triples, in barycentric coordinates."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    """Minimum triangle area as a fraction of the triangle area."""
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _combinatorial_refine(points, deadline):
    """
    Attempt to improve the minimum area by swapping points that are part of
    the minimum area triangles with other points, if it increases the min area.
    """
    bary = _cart_to_bary(points)
    best_bary = bary.copy()
    best_score = _min_area_ratio_from_bary(best_bary)
    
    # Pre-compute all triangle areas for current configuration
    dets = _dets_from_bary(best_bary)
    A = np.abs(dets)
    min_area = np.min(A)
    
    # Find indices of triangles that achieve the minimum area
    min_indices = np.where(A <= min_area * 1.000001)[0]
    
    # Points involved in these minimum triangles
    involved_points = set()
    for idx in min_indices:
        involved_points.add(_I[idx])
        involved_points.add(_J[idx])
        involved_points.add(_K[idx])
        
    # Try swapping each involved point with every other point
    # We only swap if the new min area is strictly larger
    for p_idx in list(involved_points):
        if time.time() > deadline:
            break
            
        for q_idx in range(N):
            if q_idx == p_idx:
                continue
                
            # Swap p_idx and q_idx in barycentric coordinates
            new_bary = best_bary.copy()
            new_bary[p_idx] = best_bary[q_idx]
            new_bary[q_idx] = best_bary[p_idx]
            
            # Project to simplex to ensure validity
            new_bary = _project_to_bary_simplex(new_bary)
            
            new_dets = _dets_from_bary(new_bary)
            new_A = np.abs(new_dets)
            new_min_area = np.min(new_A)
            
            if new_min_area > best_score * (1 + 1e-10):
                best_bary = new_bary
                best_score = new_min_area
                # Update min indices and involved points for next iteration
                min_indices = np.where(new_A <= new_min_area * 1.000001)[0]
                involved_points = set()
                for idx in min_indices:
                    involved_points.add(_I[idx])
                    involved_points.add(_J[idx])
                    involved_points.add(_K[idx])
                # Break out of q loop to restart p loop with new state
                break
                
    return _bary_to_cart(best_bary)

def _structured_init_fps(rng, n_init):
    """Farthest-point-sampling in barycentric coords: start from 3 vertices, then
    repeatedly add the point maximizing the minimum pairwise distance."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    for _ in range(n_init - 3):
        best_idx = -1
        best_val = -1.0
        for t in range(4000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            cand = np.array([u, v])
            d2 = np.min(np.sum((np.array(pts) - cand) ** 2, axis=1))
            if d2 > best_val:
                best_val = d2
                best_idx = cand
        pts.append(best_idx)
    return np.array(pts, dtype=float)


def _objective_smooth(x, beta):
    """Smooth (log-sum-exp) surrogate of the min twice-area, plus simplex penalty.
    Returns (value, gradient) for TRF."""
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    penalty = 0.0
    gx = np.zeros_like(x)
    lam = 100.0
    
    # Penalty for u < 0
    neg_u = -u
    pos_u = neg_u > 0.0
    if np.any(pos_u):
        penalty += lam * np.sum(neg_u[pos_u] ** 2)
        gx[0::2][pos_u] += 2.0 * lam * neg_u[pos_u]
        
    # Penalty for v < 0
    neg_v = -v
    pos_v = neg_v > 0.0
    if np.any(pos_v):
        penalty += lam * np.sum(neg_v[pos_v] ** 2)
        gx[1::2][pos_v] += 2.0 * lam * neg_v[pos_v]
        
    # Penalty for w < 0 (u+v > 1)
    over = u + v - 1.0
    pos_w = over > 0.0
    if np.any(pos_w):
        penalty += lam * np.sum(over[pos_w] ** 2)
        gx[0::2][pos_w] += 2.0 * lam * over[pos_w]
        gx[1::2][pos_w] += 2.0 * lam * over[pos_w]

    dets = _dets_from_bary(bary)
    A = np.abs(dets)
    m = np.min(A)
    shifted = A - m
    z = -beta * shifted
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsum = zmax + np.log(s)
    smooth_min = m - (1.0 / beta) * logsum

    # gradient of -smooth_min
    expz = np.exp(z - zmax)
    weights = expz / s  # softmax weights over triples
    # d smooth_min / d dets_t = -weights_t * sign(dets_t)
    gdet = -weights * np.sign(dets)
    
    gu = np.zeros(N)
    gv = np.zeros(N)
    
    # Precompute w for all points
    w_all = 1.0 - u - v
    
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = w_all[i]
        wjj = w_all[j]
        wkk = w_all[k]
        
        # Correct partials derived from:
        # det = ui(vj wk - wj vk) - vi(uj wk - wj uk) + wi(uj vk - vj uk)
        # d/dui = vj*wkk - wjj*vk
        # d/dvi = -(uj*wkk - wjj*uk)
        # d/duj = -vi*wkk + wii*vk
        # d/dvj = vi*wkk - wii*uk
        # d/duk = vi*wjj - wii*vj
        # d/dvk = ui*wjj - wjj*ui
        
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += -gd * (uj * wkk - wjj * uk)
        gu[j] += gd * (-vi * wkk + wii * vk)
        gv[j] += gd * (vi * wkk - wii * uk)
        gu[k] += gd * (vi * wjj - wii * vj)
        gv[k] += gd * (ui * wjj - wjj * ui)

    gx[0::2] += gu
    gx[1::2] += gv

    value = -smooth_min + penalty
    return value, gx


def _objective_slsqp(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def _objective_exact(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s

    rng = np.random.default_rng(1234)

    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * i + 1]})
    bounds = [(0.0, 1.0)] * (2 * N)

    best_points = None
    best_score = -float('inf')

    def consider(bary):
        nonlocal best_points, best_score
        bary = _project_to_bary_simplex(bary)
        points = _bary_to_cart(bary)
        points = _clip_to_triangle(points)
        score = _min_area_ratio_from_bary(_cart_to_bary(points))
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score

    # Phase 1: multi-start from structured FPS inits
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 15.0:
            break
        bary_init = _structured_init_fps(rng, N)
        x0 = bary_init.flatten()
        consider(bary_init)

        # 1a: TRF on smooth surrogate
        if time.time() < deadline - 10.0:
            for beta in (50.0, 150.0):
                if time.time() > deadline - 8.0:
                    break
                try:
                    res = minimize(
                        _objective_smooth, x0, method='trust-constr',
                        jac=True, bounds=bounds,
                        options={'maxiter': 150, 'gtol': 1e-8}
                    )
                    bary_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
                    consider(bary_opt)
                    x0 = bary_opt.flatten()
                except Exception:
                    pass

        # 1b: SLSQP on exact objective
        if time.time() < deadline - 8.0:
            try:
                res = minimize(_objective_slsqp, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 200, 'ftol': 1e-9})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 1c: Combinatorial Refinement
        if time.time() < deadline - 5.0:
            improved_points = _combinatorial_refine(best_points, deadline - 2.0)
            consider(_cart_to_bary(improved_points))

    # Phase 2: refine the global best
    if best_points is not None:
        bary_best = _cart_to_bary(best_points)
        x_best = bary_best.flatten()
        consider(bary_best)

        # 2a: SLSQP exact
        if time.time() < deadline - 5.0:
            try:
                res = minimize(_objective_slsqp, x_best, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 300, 'ftol': 1e-11})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 2b: Nelder-Mead on exact (Cartesian)
        if time.time() < deadline - 3.0:
            def obj_cart(x):
                pts = _clip_to_triangle(x.reshape(N, 2))
                return -_min_area_ratio_from_bary(_cart_to_bary(pts))
            try:
                res = minimize(obj_cart, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10,
                                        'fatol': 1e-12, 'adaptive': True})
                consider(_cart_to_bary(_clip_to_triangle(res.x.reshape(N, 2))))
            except Exception:
                pass
        
        # 2c: Final Combinatorial Refinement
        if time.time() < deadline - 1.0:
            improved_points = _combinatorial_refine(best_points, deadline - 0.5)
            consider(_cart_to_bary(improved_points))

    if best_points is None:
        bary_init = _structured_init_fps(rng, N)
        best_points = _bary_to_cart(bary_init)

    return _clip_to_triangle(best_points)
