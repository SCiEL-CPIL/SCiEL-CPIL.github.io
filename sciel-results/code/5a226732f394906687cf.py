import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
DEADLINE = 138.0  # Safety margin before 150s limit

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRIS = len(IDX_I)

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _soft_min_value(areas, beta):
    """Stable log-sum-exp based soft minimum."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _support_function_to_points(coeffs, n_points=14):
    """
    Convert support function coefficients to point positions on the convex boundary.
    
    h(θ) = c0 + sum_{k=1}^{K} [a_k cos(kθ) + b_k sin(kθ)]
    
    Points are placed at equally-spaced angles θ_i = 2πi/n_points.
    For each θ, the boundary point is:
      x = h(θ)cos(θ) - h'(θ)sin(θ)
      y = h(θ)sin(θ) + h'(θ)cos(θ)
    
    where h'(θ) = sum_{k=1}^{K} k[-a_k sin(kθ) + b_k cos(kθ)]
    """
    K = len(coeffs) // 2
    c0 = coeffs[0]
    a = coeffs[1:K+1]
    b = coeffs[K+1:2*K+1]
    
    # Ensure strict convexity: h + h'' > 0
    # h'' = sum_{k=1}^{K} -k^2 [a_k cos(kθ) + b_k sin(kθ)]
    # We'll check this numerically later
    
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Compute h(θ) and h'(θ)
    h = np.full(n_points, c0)
    hp = np.zeros(n_points)
    
    for k in range(1, K+1):
        h += a[k-1] * np.cos(k*theta) + b[k-1] * np.sin(k*theta)
        hp += k * (-a[k-1] * np.sin(k*theta) + b[k-1] * np.cos(k*theta))
    
    # Boundary points
    x = h * np.cos(theta) - hp * np.sin(theta)
    y = h * np.sin(theta) + hp * np.cos(theta)
    
    return np.column_stack([x, y])

def _support_function_curvature(coeffs, n_check=100):
    """Check strict convexity: h + h'' > 0 everywhere."""
    K = len(coeffs) // 2
    c0 = coeffs[0]
    a = coeffs[1:K+1]
    b = coeffs[K+1:2*K+1]
    
    theta = np.linspace(0, 2*np.pi, n_check, endpoint=False)
    
    h = np.full(n_check, c0)
    hpp = np.zeros(n_check)
    
    for k in range(1, K+1):
        h += a[k-1] * np.cos(k*theta) + b[k-1] * np.sin(k*theta)
        hpp += -k**2 * (a[k-1] * np.cos(k*theta) + b[k-1] * np.sin(k*theta))
    
    min_curv = np.min(h + hpp)
    return min_curv

def _objective_support_soft(coeffs, beta=200.0):
    """Soft objective for support function parameters."""
    if np.any(~np.isfinite(coeffs)):
        return 1e10
    
    # Check convexity
    min_curv = _support_function_curvature(coeffs)
    if min_curv < 0.05:  # Require positive curvature with margin
        return 1e10 * (1.0 - min_curv / 0.05) + 1e10
    
    pts = _support_function_to_points(coeffs)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    sm = _soft_min_value(areas, beta)
    if sm < 1e-12:
        return 1e10
    
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    return hull_area / sm

def _objective_support_hard(coeffs):
    """Hard objective: -score for support function parameters."""
    if np.any(~np.isfinite(coeffs)):
        return 1e10
    
    min_curv = _support_function_curvature(coeffs)
    if min_curv < 0.05:
        return 1e10 * (1.0 - min_curv / 0.05) + 1e10
    
    pts = _support_function_to_points(coeffs)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    s = _score(pts)
    return -s if s > 0 else 1e10

def _objective_support_soft_multi(coeffs, betas=None):
    """Multi-beta soft objective."""
    if betas is None:
        betas = [50.0, 150.0, 400.0]
    vals = []
    for b in betas:
        vals.append(_objective_support_soft(coeffs, beta=b))
    return np.mean(vals)

def _generate_support_inits():
    """Generate diverse initial support function coefficient vectors."""
    inits = []
    
    # Regular polygon (circle): only c0
    for r in [0.8, 1.0, 1.2]:
        coeffs = np.zeros(13)  # c0 + 6 harmonics × 2
        coeffs[0] = r
        inits.append(coeffs)
    
    # Slightly perturbed circles
    rng = np.random.default_rng(42)
    for seed in range(20):
        rng_local = np.random.default_rng(seed * 7 + 1)
        coeffs = np.zeros(13)
        coeffs[0] = 1.0
        # Small perturbations for k=1 to 6
        for k in range(1, 7):
            coeffs[1 + k - 1] = rng_local.normal(0, 0.1)  # a_k
            coeffs[7 + k - 1] = rng_local.normal(0, 0.1)  # b_k
        inits.append(coeffs)
    
    # Ellipse-like: dominant k=1
    for ax in [0.7, 0.8, 0.9, 1.1, 1.2]:
        for ay in [0.7, 0.8, 0.9, 1.1, 1.2]:
            coeffs = np.zeros(13)
            coeffs[0] = (ax + ay) / 2
            coeffs[1] = (ax - ay) / 2  # a_1
            # b_1 = 0 for axis-aligned
            inits.append(coeffs)
    
    # Rotated ellipses
    for ax in [0.8, 1.0, 1.2]:
        for ay in [0.7, 0.9, 1.1]:
            for rot in [0.3, 0.7, 1.2, 1.8]:
                coeffs = np.zeros(13)
                coeffs[0] = (ax + ay) / 2
                # a_1 = (ax-ay)/2 * cos(2*rot), b_1 = (ax-ay)/2 * sin(2*rot)
                delta = (ax - ay) / 2
                coeffs[1] = delta * np.cos(2 * rot)
                coeffs[7] = delta * np.sin(2 * rot)
                inits.append(coeffs)
    
    # Higher harmonic dominance
    for k_dom in [2, 3, 4]:
        for amp in [0.1, 0.2, 0.3]:
            for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
                coeffs = np.zeros(13)
                coeffs[0] = 1.0
                coeffs[1 + k_dom - 1] = amp * np.cos(phase)  # a_k
                coeffs[7 + k_dom - 1] = amp * np.sin(phase)  # b_k
                inits.append(coeffs)
    
    # Mixed harmonics
    for seed in range(15):
        rng_local = np.random.default_rng(seed * 13 + 5)
        coeffs = np.zeros(13)
        coeffs[0] = 1.0
        for k in range(1, 7):
            amp = rng_local.uniform(0.02, 0.15)
            phase = rng_local.uniform(0, 2*np.pi)
            coeffs[1 + k - 1] = amp * np.cos(phase)
            coeffs[7 + k - 1] = amp * np.sin(phase)
        inits.append(coeffs)
    
    return inits

def _normalize_points(pts):
    """Center and normalize points."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm
    return pts

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def remaining_time():
        return DEADLINE - (time.time() - start_time)

    # === PHASE 1: Differential Evolution in support function space ===
    
    def de_objective(flat):
        if np.any(~np.isfinite(flat)):
            return 1.0
        s = _objective_support_hard(flat)
        return s if s < 1e9 else 1.0

    # Bounds: c0 in [0.3, 2.0], each harmonic coeff in [-0.4, 0.4]
    bounds = [(0.3, 2.0)] + [(-0.4, 0.4)] * 12

    if remaining_time() > 45.0:
        try:
            res_de = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=60,
                popsize=15,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            if res_de.fun < 1e9:
                pts14 = _support_function_to_points(res_de.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    if remaining_time() > 30.0:
        try:
            res_de2 = differential_evolution(
                de_objective,
                bounds,
                seed=123,
                maxiter=50,
                popsize=12,
                tol=1e-6,
                mutation=(0.3, 0.8),
                recombination=0.85,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if res_de2.fun < 1e9:
                pts14 = _support_function_to_points(res_de2.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # === PHASE 2: Local refinement ===
    
    support_inits = _generate_support_inits()
    
    # Add DE results if found
    if best_points is not None:
        # Try to fit support function to best_points
        # This is complex, so we'll just add regular init
        support_inits.insert(0, np.array([1.0] + [0.0] * 12))

    for i, flat_init in enumerate(support_inits):
        if remaining_time() < 20.0:
            break
        
        # Stage A: L-BFGS-B with multi-beta soft objective
        if remaining_time() > 15.0:
            try:
                res_lbfgs = minimize(
                    _objective_support_soft_multi,
                    flat_init,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                if res_lbfgs.fun < 1e10:
                    pts14 = _support_function_to_points(res_lbfgs.x)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
                    flat_init = res_lbfgs.x
            except Exception:
                pass

        # Stage B: Nelder-Mead on hard objective
        if remaining_time() > 10.0:
            try:
                res_nm = minimize(
                    _objective_support_hard,
                    flat_init,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < 1e9:
                    pts14 = _support_function_to_points(res_nm.x)
                    cand_score = -res_nm.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage C: Powell on hard objective
        if remaining_time() > 8.0:
            try:
                res_pow = minimize(
                    _objective_support_hard,
                    flat_init,
                    method='Powell',
                    options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_pow.fun < 1e9:
                    pts14 = _support_function_to_points(res_pow.x)
                    cand_score = -res_pow.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # === PHASE 3: General (non-symmetric) local search from best ===
    if best_points is not None and remaining_time() > 15.0:
        def _objective_general(flat_points):
            pts = flat_points.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            return -_score(pts)

        def _objective_general_soft(flat_points, beta=200.0):
            pts = flat_points.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _all_areas(pts)
            if len(areas) == 0 or np.min(areas) < 1e-12:
                return 1e10
            sm = _soft_min_value(areas, beta)
            if sm < 1e-12:
                return 1e10
            hull_area = _compute_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            return hull_area / sm

        pts_norm = _normalize_points(best_points)
        flat_gen = pts_norm.flatten()

        # L-BFGS-B soft
        if remaining_time() > 12.0:
            try:
                res = minimize(
                    lambda x: _objective_general_soft(x, beta=200.0),
                    flat_gen,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
                flat_gen = res.x
            except Exception:
                pass

        # Nelder-Mead hard
        if remaining_time() > 8.0:
            try:
                res = minimize(
                    _objective_general,
                    flat_gen,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass

    # === PHASE 4: Random restarts around best ===
    if best_points is not None and remaining_time() > 10.0:
        rng2 = np.random.default_rng(77)
        for k in range(8):
            if remaining_time() < 6.0:
                break
            noise_scale = 0.02 + 0.015 * k
            noise = rng2.normal(0, noise_scale, size=(N, 2))
            perturbed = _normalize_points(best_points + noise)
            try:
                res = minimize(
                    lambda x: _objective_general_soft(x, beta=300.0),
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
