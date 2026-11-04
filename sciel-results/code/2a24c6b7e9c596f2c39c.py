import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute for fast triangle area computation
def _fast_min_triangle_area(points):
    """Compute min triangle area without hull (assumes convex position)."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    return 0.5 * np.min(np.abs(cross))

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = _fast_min_triangle_area(points)
    return min_area / hull_area

def _hull_area_cached(points):
    """Compute convex hull area, return 0.0 on failure."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    """
    hull_area = _hull_area_cached(points)
    if hull_area <= 1e-9:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    abs_cross = np.abs(cross)
    
    # Soft min via log-sum-exp for numerical stability
    max_val = np.max(abs_cross)
    exp_vals = np.exp(-beta * (abs_cross - max_val))
    soft_min_val = max_val + np.log(np.sum(exp_vals)) / beta
    
    return 0.5 * soft_min_val / hull_area

def _fourier_ring_points(n_outer, n_inner, fourier_coeffs, r_inner, theta_inner):
    """
    Generate points using Fourier series for outer ring radius.
    fourier_coeffs: array of shape (n_outer+1,) where index 0 is mean radius,
                   odd indices are cosine amplitudes, even indices (>=2) are sine amplitudes.
    Actually, let's use: coeffs[0]=r0 (mean), coeffs[1:]=Fourier coefficients for k=1..n_outer-1
    r_k = r0 + sum_{m=1}^{M} (a_m * cos(m*theta_k) + b_m * sin(m*theta_k))
    """
    n_total = n_outer + n_inner
    M = min(n_outer - 1, len(fourier_coeffs) - 1)
    
    angles_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    
    # Compute radius for each outer point
    r_out = np.zeros(n_outer)
    r_out[:] = fourier_coeffs[0]
    for m in range(1, M + 1):
        a_m = fourier_coeffs[2*m - 1] if 2*m - 1 < len(fourier_coeffs) else 0.0
        b_m = fourier_coeffs[2*m] if 2*m < len(fourier_coeffs) else 0.0
        r_out += a_m * np.cos(m * angles_out) + b_m * np.sin(m * angles_out)
    
    x_out = r_out * np.cos(angles_out)
    y_out = r_out * np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + theta_inner
    x_in = r_inner * np.cos(angles_in)
    y_in = r_inner * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return points

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta] for simple ring, or extended for Fourier.
    For 2-param: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _score_fourier_params(n_out, n_inner, params):
    """
    Score for Fourier-ring parameterization.
    params: [r0, a1, b1, a2, b2, ..., r_inner, theta_inner]
    """
    n_coeffs = 2 * (n_out - 1) + 1  # r0 + (a_m, b_m) for m=1..n_out-1
    r_inner = params[n_coeffs]
    theta_inner = params[n_coeffs + 1]
    
    if r_inner < 0 or r_inner > 0.999:
        return 0.0
    
    points = _fourier_ring_points(n_out, n_inner, params[:n_coeffs], r_inner, theta_inner)
    return _calculate_score(points)

def _soft_fourier_params(n_out, n_inner, params, beta=80.0):
    """
    Soft score for Fourier-ring (negative for minimization).
    """
    n_coeffs = 2 * (n_out - 1) + 1
    r_inner = params[n_coeffs]
    theta_inner = params[n_coeffs + 1]
    
    if r_inner < 0 or r_inner > 0.999:
        return 1.0
    
    points = _fourier_ring_points(n_out, n_inner, params[:n_coeffs], r_inner, theta_inner)
    return -_soft_min_score(points, beta)

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
    """
    Smooth objective for gradient-based optimization (minimize negative).
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 1.0
        
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return -_soft_min_score(points, beta)

def _optimize_fourier_ring(n_out, n_in, time_limit):
    """
    Optimize a Fourier-ring configuration with asymmetric outer ring.
    """
    start_time = time.time()
    
    n_coeffs = 2 * (n_out - 1) + 1
    total_dim = n_coeffs + 2  # + r_inner, theta_inner
    
    # Bounds: r0 in [0.7, 1.1], Fourier coeffs small, r_inner in [0, 0.999], theta in [0, 2pi]
    bounds = [(0.7, 1.1)]  # r0
    for m in range(1, n_out):
        bounds.append((-0.15, 0.15))  # a_m
        bounds.append((-0.15, 0.15))  # b_m
    bounds.append((0.0, 0.999))   # r_inner
    bounds.append((0.0, 2*np.pi)) # theta_inner
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: DE for global exploration (only on a subset of params to keep it tractable)
    # Use DE on all params but with limited iterations
    try:
        popsize = 12
        maxiter = 30 if total_dim <= 15 else 20
        
        res = differential_evolution(
            lambda p: -_score_fourier_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _score_fourier_params(n_out, n_in, res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Also try starting from symmetric ring (all Fourier coeffs = 0)
    sym_params = np.zeros(total_dim)
    sym_params[0] = 1.0  # r0 = 1
    sym_params[n_coeffs] = 0.5  # r_inner
    sym_params[n_coeffs + 1] = 0.0  # theta
    
    sym_score = _score_fourier_params(n_out, n_in, sym_params)
    if sym_score > best_score:
        best_score = sym_score
        best_params = sym_params.copy()
    
    # Phase 2: Local polish with L-BFGS-B on soft objective
    if time.time() - start_time < time_limit * 0.7:
        starts = []
        if best_params is not None:
            starts.append(best_params.copy())
        starts.append(sym_params.copy())
        
        # Add perturbed starts
        for start in list(starts):
            for delta in [0.02, -0.02, 0.05, -0.05]:
                perturbed = start.copy()
                perturbed[1] += delta  # perturb a1
                if 0.0 <= perturbed[n_coeffs] <= 0.999:
                    starts.append(perturbed)
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.7:
                break
            try:
                res_lbfgs = minimize(
                    lambda p: _soft_fourier_params(n_out, n_in, p, beta=80.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if res_lbfgs.x is not None:
                    exact_score = _score_fourier_params(n_out, n_in, res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish
        if best_params is not None and time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_score_fourier_params(n_out, n_in, p),
                    best_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 600, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return 0.0, None
    
    return best_score, best_params

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: DE for global exploration
    try:
        popsize = 15
        maxiter = 60 if (n_out, n_in) == (7, 7) else 40
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _score_ring_params(n_out, n_in, res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish with smooth surrogate + L-BFGS-B
    if time.time() - start_time < time_limit * 0.7:
        starts = []
        
        if best_params is not None:
            starts.append(best_params.copy())
            
        r_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        theta_values = [0.0, theta_max/4, theta_max/2, 3*theta_max/4]
        
        for r_start in r_values:
            for theta_start in theta_values:
                starts.append([r_start, theta_start])
        
        if best_params is not None:
            for dr in [-0.05, 0.05]:
                for dtheta in [-0.1, 0.1]:
                    r_p = np.clip(best_params[0] + dr, 0.01, 0.998)
                    th_p = (best_params[1] + dtheta) % theta_max
                    starts.append([r_p, th_p])
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.7:
                break
            try:
                res_lbfgs = minimize(
                    lambda p: _soft_score_ring_params(n_out, n_in, p, beta=100.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                
                if res_lbfgs.x is not None:
                    exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final Nelder-Mead polish on best point
        if best_params is not None and time.time() - start_time < time_limit:
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    best_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-11, 'fatol': 1e-11, 'maxiter': 800, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _polish_points(points, time_limit):
    """
    Direct coordinate polish on all points using soft-min objective.
    Multiple restarts with different beta values for robustness.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Use L-BFGS-B with soft-min objective on all 28 coordinates
    # Try multiple beta values for different smoothness levels
    for beta in [30.0, 60.0, 100.0]:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        flat = best_points.flatten()
        
        def obj(x, b=beta):
            pts = x.reshape(-1, 2)
            return -_soft_min_score(pts, b)
        
        try:
            res = minimize(
                obj,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                new_pts = res.x.reshape(-1, 2)
                new_score = _calculate_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts
        except Exception:
            pass
    
    # Nelder-Mead polish on best point
    if time.time() - start_time < time_limit * 0.5:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-11, 'fatol': 1e-11, 'maxiter': 1200, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Final tight polish with very small steps
    if time.time() - start_time < time_limit:
        try:
            res_nm2 = minimize(
                lambda x: -_calculate_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-13, 'fatol': 1e-13, 'maxiter': 800, 'adaptive': True,
                         'initial_simplex': None}
            )
            if -res_nm2.fun > best_score:
                best_score = -res_nm2.fun
                best_points = res_nm2.x.reshape(-1, 2)
        except Exception:
            pass
    
    return best_score, best_points

def _three_ring_score(params, n1, n2, n3):
    """
    Three-ring configuration: outer unit circle, middle ring at r1, inner ring at r2.
    params: [r1, r2, theta1, theta2]
    """
    r1, r2, theta1, theta2 = params
    if r1 < 0 or r1 > 0.999 or r2 < 0 or r2 > 0.999:
        return 0.0
    
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    x1 = np.cos(angles1)
    y1 = np.sin(angles1)
    
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta1
    x2 = r1 * np.cos(angles2)
    y2 = r1 * np.sin(angles2)
    
    angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + theta2
    x3 = r2 * np.cos(angles3)
    y3 = r2 * np.sin(angles3)
    
    points = np.vstack([
        np.column_stack((x1, y1)),
        np.column_stack((x2, y2)),
        np.column_stack((x3, y3))
    ])
    
    return _calculate_score(points)

def _optimize_three_ring(n1, n2, n3, time_limit):
    """
    Optimize a three-ring configuration.
    """
    start_time = time.time()
    
    bounds = [(0.0, 0.999), (0.0, 0.999), (0.0, 2*np.pi), (0.0, 2*np.pi)]
    
    best_score = 0.0
    best_params = None
    
    # DE for global exploration
    try:
        res = differential_evolution(
            lambda p: -_three_ring_score(p, n1, n2, n3),
            bounds,
            seed=42,
            maxiter=30,
            popsize=12,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _three_ring_score(res.x, n1, n2, n3)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Local polish
    if best_params is not None and time.time() - start_time < time_limit * 0.7:
        try:
            res_nm = minimize(
                lambda p: -_three_ring_score(p, n1, n2, n3),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

# --- Lattice-based approach ---

def _lattice_score(params, lattice_points):
    """
    Score for a lattice point set transformed by affine parameters.
    params: [sx, sy, theta, shear_x, shear_y, tx, ty]
    lattice_points: (N, 2) integer grid points
    """
    sx, sy, theta, shx, shy, tx, ty = params
    
    # Build transformation matrix
    # Rotation
    c, s = np.cos(theta), np.sin(theta)
    # Shear
    # Scale
    # Combined: Scale * Shear * Rotation
    
    # Actually, let's define the affine map as:
    # p' = A * p + t
    # where A = [[sx, 0], [0, sy]] * [[1, shx], [shy, 1]] * [[c, -s], [s, c]]
    
    # Rotation matrix R
    R = np.array([[c, -s], [s, c]])
    # Shear matrix S
    S = np.array([[1, shx], [shy, 1]])
    # Scale matrix K
    K = np.array([[sx, 0], [0, sy]])
    
    A = K @ S @ R
    
    transformed = lattice_points @ A.T + np.array([tx, ty])
    
    return _calculate_score(transformed)

def _soft_lattice_score(params, lattice_points, beta=80.0):
    """
    Soft score for lattice approach.
    """
    sx, sy, theta, shx, shy, tx, ty = params
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = np.array([[1, shx], [shy, 1]])
    K = np.array([[sx, 0], [0, sy]])
    
    A = K @ S @ R
    transformed = lattice_points @ A.T + np.array([tx, ty])
    
    return -_soft_min_score(transformed, beta)

def _generate_lattice_configs():
    """
    Generate several lattice point configurations with 14 points.
    Returns list of (N, 2) integer arrays.
    """
    configs = []
    
    # 3x5 grid with corners removed: 15 - 1 = 14? No, 3x5=15. Remove 1 corner.
    # Actually, let's try various subsets of small grids.
    
    # Config 1: 4x4 grid minus 2 points = 14
    # Full 4x4 is 16 points. Remove 2.
    grid_4x4 = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    # Remove (0,0) and (3,3) for symmetry
    mask = ~((grid_4x4[:, 0] == 0) & (grid_4x4[:, 1] == 0)) & ~((grid_4x4[:, 0] == 3) & (grid_4x4[:, 1] == 3))
    configs.append(grid_4x4[mask])
    
    # Config 2: 3x5 grid minus 1 point = 14
    grid_3x5 = np.array([(i, j) for i in range(3) for j in range(5)], dtype=float)
    # Remove center-ish point? Or a corner. Let's remove (1, 2)
    mask = ~((grid_3x5[:, 0] == 1) & (grid_3x5[:, 1] == 2))
    configs.append(grid_3x5[mask])
    
    # Config 3: 2x7 grid = 14 points
    grid_2x7 = np.array([(i, j) for i in range(2) for j in range(7)], dtype=float)
    configs.append(grid_2x7)
    
    # Config 4: 5x3 grid minus 1
    grid_5x3 = np.array([(i, j) for i in range(5) for j in range(3)], dtype=float)
    mask = ~((grid_5x3[:, 0] == 2) & (grid_5x3[:, 1] == 1))
    configs.append(grid_5x3[mask])
    
    # Config 5: Hexagonal-like packing
    # Rows of 5, 4, 5 = 14
    hex_pts = []
    for row, count in enumerate([5, 4, 5]):
        offset = 0.5 if row % 2 == 1 else 0.0
        for col in range(count):
            hex_pts.append((col + offset, row * 0.866))
    hex_pts = np.array(hex_pts)
    configs.append(hex_pts)
    
    # Config 6: Another hexagonal variant
    hex_pts2 = []
    for row, count in enumerate([4, 5, 5]):
        offset = 0.5 if row % 2 == 1 else 0.0
        for col in range(count):
            hex_pts2.append((col + offset, row * 0.866))
    hex_pts2 = np.array(hex_pts2)
    configs.append(hex_pts2)
    
    return configs

def _optimize_lattice(lattice_points, time_limit):
    """
    Optimize affine transformation for a given lattice point set.
    """
    start_time = time.time()
    
    # Parameters: [sx, sy, theta, shx, shy, tx, ty]
    # Initial guess: identity transform, centered
    center = np.mean(lattice_points, axis=0)
    lattice_centered = lattice_points - center
    
    bounds = [
        (0.1, 3.0),   # sx
        (0.1, 3.0),   # sy
        (0.0, 2*np.pi), # theta
        (-0.5, 0.5),  # shx
        (-0.5, 0.5),  # shy
        (-1.0, 1.0),  # tx
        (-1.0, 1.0)   # ty
    ]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: DE for global exploration
    try:
        initial_guess = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        res = differential_evolution(
            lambda p: -_lattice_score(p, lattice_points),
            bounds,
            seed=42,
            maxiter=20,
            popsize=10,
            tol=1e-4,
            polish=False,
            x0=initial_guess
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x.copy()
        elif hasattr(res, 'x') and res.x is not None:
            s = _lattice_score(res.x, lattice_points)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish
    if best_params is not None and time.time() - start_time < time_limit * 0.6:
        try:
            res_nm = minimize(
                lambda p: -_lattice_score(p, lattice_points),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, use 145s
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Phase 1: Standard ring optimization for top candidates
    ring_candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
    ]
    
    for n_out, n_in in ring_candidates:
        if time.time() > deadline - 30:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        else:
            t_limit = 10.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                x_out = np.cos(angles_out)
                y_out = np.sin(angles_out)
                
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                x_in = r * np.cos(angles_in)
                y_in = r * np.sin(angles_in)
                
                best_overall_points = np.vstack([
                    np.column_stack((x_out, y_out)),
                    np.column_stack((x_in, y_in))
                ])
    
    # Phase 2: Fourier-ring optimization (asymmetric outer ring)
    fourier_configs = [(7, 7), (8, 6), (6, 8)]
    for n_out, n_in in fourier_configs:
        if time.time() > deadline - 20:
            break
        
        t_limit = min(25.0, deadline - 20 - time.time())
        if t_limit < 5:
            break
        
        score, params = _optimize_fourier_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                n_coeffs = 2 * (n_out - 1) + 1
                r_inner = params[n_coeffs]
                theta_inner = params[n_coeffs + 1]
                best_overall_points = _fourier_ring_points(
                    n_out, n_in, params[:n_coeffs], r_inner, theta_inner
                )
    
    # Phase 3: Three-ring configurations
    three_ring_configs = [(5, 5, 4), (6, 4, 4), (7, 3, 4)]
    for n1, n2, n3 in three_ring_configs:
        if time.time() > deadline - 10:
            break
        if n1 + n2 + n3 != N:
            continue
            
        t_limit = min(8.0, deadline - 10 - time.time())
        if t_limit < 3:
            break
            
        score, params = _optimize_three_ring(n1, n2, n3, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r1, r2, theta1, theta2 = params
                
                angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
                x1 = np.cos(angles1)
                y1 = np.sin(angles1)
                
                angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + theta1
                x2 = r1 * np.cos(angles2)
                y2 = r1 * np.sin(angles2)
                
                angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + theta2
                x3 = r2 * np.cos(angles3)
                y3 = r2 * np.sin(angles3)
                
                best_overall_points = np.vstack([
                    np.column_stack((x1, y1)),
                    np.column_stack((x2, y2)),
                    np.column_stack((x3, y3))
                ])
    
    # Phase 4: Final polish on best points using direct coordinate optimization
    if best_overall_points is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 5:
            polished_score, polished_points = _polish_points(best_overall_points, remaining)
            if polished_score > best_overall_score:
                best_overall_score = polished_score
                best_overall_points = polished_points
    
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        x_out = np.cos(angles_out)
        y_out = np.sin(angles_out)
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
        x_in = r * np.cos(angles_in)
        y_in = r * np.sin(angles_in)
        best_overall_points = np.vstack([
            np.column_stack((x_out, y_out)),
            np.column_stack((x_in, y_in))
        ])
        
    return best_overall_points
