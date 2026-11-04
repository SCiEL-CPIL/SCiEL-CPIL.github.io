import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
_N_TRIPLES = len(_IDX_TRIPLES)

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _param_to_points(params):
    """Map parameters in [0, 1]^{2N} to points inside the triangle."""
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
    """Map points in the triangle to parameters (s, t)."""
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

def _soft_min(areas, beta):
    """Smooth approximation of min using log-sum-exp."""
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return -log_sum / beta

def _objective_smooth(params, beta):
    """Smooth objective: negative soft-min of triangle areas."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return -_soft_min(areas, beta)

def _objective(params):
    """Objective: negative min triangle area (hard min)."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """Generate an initial configuration using Greedy Farthest Point Sampling."""
    points = np.zeros((n_points, 2))
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

def _structured_init(rng, seed_offset=0):
    """Generate a structured initial configuration based on grid-like placement."""
    rng_local = np.random.RandomState(42 + seed_offset)
    points = np.zeros((N, 2))
    
    # Try different structured patterns
    pattern = seed_offset % 4
    
    if pattern == 0:
        # Grid-like pattern
        rows = np.linspace(0.15, 0.85, 4)
        cols_per_row = [3, 3, 3, 2]
        idx = 0
        for r_i, row_y in enumerate(rows):
            n_cols = cols_per_row[r_i]
            # Adjust x range based on y (triangle constraint)
            max_x = 1.0 - row_y * 2.0 / np.sqrt(3)
            min_x = row_y * 1.0 / np.sqrt(3)
            if n_cols > 1:
                cols = np.linspace(min_x + 0.05, max_x - 0.05, n_cols)
            else:
                cols = [(min_x + max_x) / 2.0]
            for c in cols:
                if idx < N:
                    points[idx] = [c, row_y]
                    idx += 1
    elif pattern == 1:
        # Hexagonal-like pattern
        centers = [
            [0.5, 0.577],  # center
            [0.25, 0.289], [0.75, 0.289],  # lower
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],  # bottom
            [0.25, 0.577], [0.75, 0.577],  # upper sides
            [0.5, 0.866],  # top
            [0.1, 0.1], [0.9, 0.1]
        ]
        for i in range(N):
            if i < len(centers):
                points[i] = centers[i]
    elif pattern == 2:
        # Fibonacci-like spiral
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(N):
            angle = i * 2.399963  # golden angle in radians
            r = 0.45 * np.sqrt((i + 0.5) / N)
            x = 0.5 + r * np.cos(angle)
            y = 0.289 + r * np.sin(angle) * 0.866
            # Clip to triangle
            if y < 0:
                y = 0.01
            if x < 0:
                x = 0.01
            if x > 1:
                x = 0.99
            # Check if inside triangle
            if y > (1 - x) * np.sqrt(3):
                y = (1 - x) * np.sqrt(3) * 0.95
            points[i] = [x, y]
    else:
        # Concentric rings
        ring_radii = [0.0, 0.25, 0.45]
        points_per_ring = [1, 4, 6]
        idx = 0
        for r_i, radius in enumerate(ring_radii):
            n_pts = points_per_ring[r_i]
            if n_pts == 1:
                points[idx] = [0.5, 0.289]
                idx += 1
            else:
                for j in range(n_pts):
                    angle = 2 * np.pi * j / n_pts
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.289 + radius * np.sin(angle) * 0.866
                    if 0 <= x <= 1 and y >= 0 and y <= (1 - x) * np.sqrt(3):
                        points[idx] = [x, y]
                        idx += 1
                    if idx >= N:
                        break
            if idx >= N:
                break
        # Fill remaining with random
        while idx < N:
            r1 = rng_local.uniform(0, 1)
            r2 = rng_local.uniform(0, 1)
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            points[idx] = [r1 + 0.5 * r2, 0.866 * r2]
            idx += 1
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 12
    for i in range(n_fps_inits):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
    
    # Add structured initializations
    for i in range(6):
        if time_left() < 40.0:
            break
        pts = _structured_init(rng, seed_offset=i)
        params = _points_to_param(pts)
        initial_params.append(params)
    
    # Add random initializations
    for i in range(8):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Multi-stage optimization:
    # Stage 1: Smooth objective with low beta (broad search)
    # Stage 2: Smooth objective with high beta (sharper)
    # Stage 3: Hard min with SLSQP
    # Stage 4: Nelder-Mead polish
    
    betas = [5.0, 20.0, 50.0, 100.0]
    
    for start_p in initial_params:
        if time_left() < 25.0:
            break
            
        current_p = start_p.copy()
        
        try:
            # Stage 1 & 2: Smooth optimization with increasing beta
            for beta in betas[:2]:
                if time_left() < 20.0:
                    break
                res = minimize(
                    lambda p: _objective_smooth(p, beta),
                    current_p, method='SLSQP', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8}
                )
                if res.success or res.fun < _objective_smooth(current_p, beta):
                    current_p = res.x.copy()
            
            # Stage 3: Hard min with SLSQP
            if time_left() > 15.0:
                res = minimize(
                    _objective, current_p, method='SLSQP', bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
                    current_p = res.x.copy()
            
            # Stage 4: Nelder-Mead polish
            if time_left() > 10.0:
                res = minimize(
                    _objective, current_p, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
        except Exception:
            pass

    # 3. Final refinement: try more SLSQP from perturbed best
    if best_params is not None and time_left() > 30.0:
        for i in range(8):
            if time_left() < 15.0:
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
    
    # 4. Final Nelder-Mead polish with very tight tolerances
    if best_params is not None and time_left() > 15.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    return np.array(points)
