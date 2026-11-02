"""Heilbronn 11 triangles: 2D Barycentric optimization with DE exploration."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _uv_to_cart(uv_flat):
    """
    Convert flat array of 2D barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2 = v*(1,0) + w*(0.5, S3H) = (v + 0.5*w, w*S3H)
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_uv(pts):
    """
    Convert Cartesian points to 2D barycentric coords (u, v) where w = 1-u-v.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _project_to_simplex(uv_flat):
    """Project 2D barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        if u + v > 1.0:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


def _uv_objective(uv_flat, beta):
    """
    Objective function for L-BFGS-B in 2D barycentric coordinates.
    Minimizes negative soft-min area with penalty for simplex violation.
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for w < 0 (i.e., u+v > 1)
    viol = np.maximum(0.0, -w)
    penalty = 500.0 * np.sum(viol ** 2)
    
    pts = _uv_to_cart(uv_flat)
    ma = _soft_min_area(pts, beta)
    return -ma + penalty


def _hard_uv_objective(uv_flat):
    """Hard objective in 2D barycentric coordinates."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    viol = np.maximum(0.0, -w)
    penalty = 500.0 * np.sum(viol ** 2)
    pts = _uv_to_cart(uv_flat)
    return -_min_triangle_area(pts) + penalty


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid(pts):
            return
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    def _generate_initial_uvs():
        """Generate diverse starting points in 2D barycentric coordinates."""
        configs = []
        
        # 1. Symmetric base: Corners, midpoints, centroid, etc.
        base_sym_cart = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        base_uv = _cart_to_uv(base_sym_cart)
        configs.append(base_uv.flatten())

        # 2. Random valid barycentric configurations
        for _ in range(12):
            # Generate random points in simplex
            raw = rng.random((N, 3))
            bary = raw / raw.sum(axis=1, keepdims=True)
            u = bary[:, 0]
            v = bary[:, 1]
            configs.append(np.column_stack([u, v]).flatten())

        # 3. Perturbed symmetric base
        for _ in range(8):
            perturbed = base_uv + rng.normal(0, 0.05, size=(N, 2))
            perturbed = _project_to_simplex(perturbed)
            configs.append(perturbed)

        # 4. More aggressive random starts
        for _ in range(6):
            # Uniform random in Cartesian, then convert
            raw_pts = rng.random((N, 2))
            # Map to triangle: use barycentric random
            r1 = rng.random(N)
            r2 = rng.random(N)
            # Acceptance-rejection style: generate in unit square, map to triangle
            pts_cart = np.zeros((N, 2))
            for i in range(N):
                u = r1[i]
                v = r2[i] * (1 - u)
                x = v + 0.5 * (1 - u - v)
                y = (1 - u - v) * S3H
                pts_cart[i] = [x, y]
            uv = _cart_to_uv(pts_cart)
            configs.append(uv.flatten())

        return configs

    def _differential_evolution(uv0, pop_size=15, n_gen=100, F=0.8, CR=0.7):
        """Differential Evolution in 2D barycentric space."""
        # Initialize population: seed with uv0, rest random
        pop = np.zeros((pop_size, N, 2))
        pop[0] = uv0
        for i in range(1, pop_size):
            raw = rng.random((N, 3))
            bary = raw / raw.sum(axis=1, keepdims=True)
            pop[i] = np.column_stack([bary[:, 0], bary[:, 1]])
        
        # Evaluate initial population
        pop_areas = np.zeros(pop_size)
        for i in range(pop_size):
            pts = _uv_to_cart(pop[i].flatten())
            pop_areas[i] = _min_triangle_area(pts)
        
        best_idx = np.argmax(pop_areas)
        best_uv = pop[best_idx].copy()
        best_area = pop_areas[best_idx]
        
        for gen in range(n_gen):
            if not _elapsed_ok(5):
                break
            
            # Sort population by fitness (descending for maximization)
            order = np.argsort(-pop_areas)
            sorted_pop = pop[order]
            sorted_areas = pop_areas[order]
            
            # Generate offspring for each member
            for i in range(pop_size):
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # DE/rand/1/bin mutation
                v = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
                
                # Project to simplex
                v = _project_to_simplex(v.flatten()).reshape(N, 2)
                
                # Binary crossover
                trial = np.where(rng.random((N, 2)) < CR, v, sorted_pop[i])
                trial = _project_to_simplex(trial.flatten()).reshape(N, 2)
                
                # Evaluate trial
                trial_pts = _uv_to_cart(trial.flatten())
                trial_area = _min_triangle_area(trial_pts)
                
                # Selection: keep better of parent and trial
                if trial_area >= sorted_areas[i]:
                    pop[i] = trial
                    pop_areas[i] = trial_area
                    if trial_area > best_area:
                        best_area = trial_area
                        best_uv = trial.copy()
            
            # Reorder population for next generation
            order = np.argsort(-pop_areas)
            pop = pop[order]
            pop_areas = pop_areas[order]
        
        return best_uv, best_area

    def _optimize_single_uv(uv0, beta_schedule):
        """Run L-BFGS-B with a schedule of beta values in 2D barycentric space."""
        current_uv = uv0.copy()
        
        for beta in beta_schedule:
            if not _elapsed_ok(8):
                break
                
            try:
                bounds = [(0, 1), (0, 1)] * N
                res = minimize(
                    _uv_objective,
                    current_uv,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_uv = res.x
                current_uv = _project_to_simplex(current_uv)
                pts = _uv_to_cart(current_uv)
                _update_best(pts)
            except Exception:
                pass
                
        return current_uv

    # Phase 1: Multi-start optimization with annealing
    configs = _generate_initial_uvs()
    
    # Beta schedule: start smooth, get sharp
    beta_schedule = [10, 30, 80, 200, 500]

    for config in configs:
        if not _elapsed_ok(15):
            break
        _optimize_single_uv(config, beta_schedule)

    # Phase 2: Differential Evolution on the best candidate
    if best_pts is not None and _elapsed_ok(25):
        best_uv = _cart_to_uv(best_pts)
        # Run DE with multiple configurations
        for de_iter in range(3):
            if not _elapsed_ok(12):
                break
            de_uv, de_area = _differential_evolution(
                best_uv,
                pop_size=15,
                n_gen=80,
                F=0.8,
                CR=0.7
            )
            de_pts = _uv_to_cart(de_uv.flatten())
            _update_best(de_pts)
            if _elapsed_ok(5):
                # Polish with L-BFGS
                uv_polished = _optimize_single_uv(de_uv, [300, 800])
                pts_polished = _uv_to_cart(uv_polished)
                _update_best(pts_polished)

    # Phase 3: Powell Polish on the best candidate (in Cartesian space for robustness)
    if best_pts is not None and _elapsed_ok(15):
        def powell_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                powell_obj,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 150, 'maxfev': 8000, 'xrtol': 1e-10, 'fatol': 1e-12}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    # Phase 4: Final fine polish with Nelder-Mead on the absolute best
    if best_pts is not None and _elapsed_ok(8):
        def nm_obj_final(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj_final,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    # Phase 5: Additional DE rounds if time permits
    if best_pts is not None and _elapsed_ok(10):
        best_uv = _cart_to_uv(best_pts)
        for de_iter in range(2):
            if not _elapsed_ok(5):
                break
            de_uv, de_area = _differential_evolution(
                best_uv,
                pop_size=12,
                n_gen=50,
                F=0.6,
                CR=0.5
            )
            de_pts = _uv_to_cart(de_uv.flatten())
            _update_best(de_pts)
            if _elapsed_ok(3):
                uv_polished = _optimize_single_uv(de_uv, [500, 1000])
                pts_polished = _uv_to_cart(uv_polished)
                _update_best(pts_polished)

    if best_pts is None:
        # Absolute fallback: random points
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
