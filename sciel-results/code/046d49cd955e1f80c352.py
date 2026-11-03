"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area.
Combined approach: vectorized barycentric area computation + smooth min objective + multi-start local optimization.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])


def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))


def _all_dets_from_bary(bary):
    """
    Compute all determinants (twice the area in barycentric coords) for all triples.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det


def _min_area_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))


def _objective_smooth(x, beta=50.0, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    neg_u = np.maximum(0, -u)
    neg_v = np.maximum(0, -v)
    penalty += penalty_lambda * (np.sum(neg_u**2) + np.sum(neg_v**2))
    
    # Clamp for area computation
    u_clamped = np.clip(u, 0, None)
    v_clamped = np.clip(v, 0, None)
    s = u_clamped + v_clamped
    mask = s > 1.0
    if np.any(mask):
        u_clamped[mask] /= s[mask]
        v_clamped[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u_clamped, v_clamped))
    dets = _all_dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min approximation
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty


def _objective_exact(x):
    """
    Exact objective for SLSQP/Nelder-Mead: minimize -min_area.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area


def _project_to_simplex(bary):
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


def _generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(20):
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
        
    # 3. Structured: Hexagonal lattice pattern
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
    inits.append(_project_to_simplex(bary1).flatten())
    
    # 4. Structured: Concentric hexagons
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
    inits.append(_project_to_simplex(bary2).flatten())
    
    # 5. Symmetric grid with jitter
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
    inits.append(_project_to_simplex(bary3).flatten())
    
    # 6. Random with structured bias
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
    inits.append(_project_to_simplex(np.array(bary4[:N])).flatten())
    
    return inits


def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0 with smooth objective."""
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_exact, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_exact, x0, method='SLSQP',
            bounds=[(0.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = _generate_inits()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: L-BFGS-B with smooth objective from all inits
    for init in inits:
        if time_left() < 5:
            break
        
        x_opt = _local_optimize_lbfgs(init, maxiter=150, gtol=1e-8)
        bary_opt = _project_to_simplex(x_opt.reshape(N, 2))
        score = _min_area_from_bary(bary_opt)
        
        if score > best_score:
            best_score = score
            best_x = bary_opt.flatten()
    
    # Phase 2: Nelder-Mead refinement from top candidates
    if time_left() > 10:
        # Take the best few solutions and refine with NM
        candidates = [best_x] if best_x is not None else []
        
        # Also add some random perturbations of the best
        if best_x is not None:
            rng_refine = np.random.RandomState(789)
            for _ in range(5):
                perturbed = best_x + rng_refine.normal(0, 0.01, size=2*N)
                perturbed_bary = _project_to_simplex(perturbed.reshape(N, 2))
                candidates.append(perturbed_bary.flatten())
        
        for cand in candidates:
            if time_left() < 5:
                break
            
            x_nm = _local_optimize_nm(cand, maxiter=200, xatol=1e-9, fatol=1e-9)
            bary_nm = _project_to_simplex(x_nm.reshape(N, 2))
            score = _min_area_from_bary(bary_nm)
            
            if score > best_score:
                best_score = score
                best_x = bary_nm.flatten()
    
    # Phase 3: SLSQP refinement if time permits
    if time_left() > 10 and best_x is not None:
        x_slsqp = _local_optimize_slsqp(best_x, maxiter=150)
        bary_slsqp = _project_to_simplex(x_slsqp.reshape(N, 2))
        score = _min_area_from_bary(bary_slsqp)
        
        if score > best_score:
            best_score = score
            best_x = bary_slsqp.flatten()
    
    # If no valid solution found, use FPS fallback
    if best_x is None:
        best_x = _structured_init_fps().flatten()
    
    # Convert to Cartesian
    points = _bary_to_cart(best_x.reshape(N, 2))
    
    # Ensure points are inside the triangle
    for i in range(N):
        x, y = points[i]
        v_b = 2.0 * y / np.sqrt(3)
        u_b = x - 0.5 * v_b
        w_b = 1.0 - u_b - v_b
        if u_b < -1e-9 or v_b < -1e-9 or w_b < -1e-9:
            cx = 0.5
            cy = np.sqrt(3) / 6.0
            points[i] = [cx + 0.9 * (x - cx), cy + 0.9 * (y - cy)]
    
    return points
