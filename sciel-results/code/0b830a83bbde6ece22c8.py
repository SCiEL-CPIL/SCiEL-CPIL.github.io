"""
Heilbronn convex regions n=14

Strategy:
1. Use a structured parameterization: points on a regular 14-gon with radial
   perturbations (r_i) and angular perturbations (d_theta_i). This keeps the
   points in convex position (or near it) and reduces the search space.

2. Use a multi-start approach with Simulated Annealing to escape local minima,
   followed by L-BFGS polishing with a smooth surrogate (soft-min of triangle
   areas).

3. The objective is to maximize min_triangle_area / hull_area.
   We minimize the negative of this, or equivalently minimize hull_area / min_triangle_area.

4. For the smooth surrogate, we use the log-sum-exp trick to approximate the
   minimum of triangle areas, which is differentiable.

5. We use multiple initial configurations: regular polygon, two-ring, and
   perturbed polygons with different seeds.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3 = 364)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _min_triangle_area(points):
    """Compute minimum triangle area."""
    areas = _all_triangle_areas(points)
    return float(np.min(areas))


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=100.0):
    """
    Soft objective: hull_area / soft_min(triangle areas).
    Uses log-sum-exp for numerical stability.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _soft_obj_grad(pts_flat, beta=100.0):
    """
    Soft objective with numerical gradient for L-BFGS-B.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, None
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, None
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10, None
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, None
    val = hull_area / soft_min_val
    
    # Numerical gradient
    grad = np.zeros_like(pts_flat)
    eps = 1e-7
    for i in range(len(pts_flat)):
        orig = pts_flat[i]
        pts_flat[i] = orig + eps
        f_plus = _soft_obj(pts_flat, beta)
        pts_flat[i] = orig - eps
        f_minus = _soft_obj(pts_flat, beta)
        pts_flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return val, grad


def _polar_to_cartesian(params):
    """
    Convert polar parameters to Cartesian points.
    params: [r_1...r_N, theta_1, d_1...d_{N-1}]
    theta_k = theta_1 + sum(d_1...d_{k-1})
    This ensures points are in angular order.
    """
    n = N
    radii = params[:n]
    theta_start = params[n]
    d_thetas = params[n+1:n+n-1]
    
    thetas = np.zeros(n)
    thetas[0] = theta_start
    if n > 1:
        thetas[1:] = theta_start + np.cumsum(d_thetas)
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])


def _neg_score_polar(params):
    """
    Objective function for optimization in polar parameter space.
    Minimizes negative score.
    """
    try:
        pts = _polar_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = _score(pts)
        return -s
    except Exception:
        return 1e10


def _soft_obj_polar(params, beta=100.0):
    """
    Soft objective in polar parameter space for L-BFGS.
    """
    try:
        pts = _polar_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e10, None
        val = _soft_obj(pts.flatten(), beta)
        return val, None
    except Exception:
        return 1e10, None


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_perturbed_polygon(n, seed=42):
    """Regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.05, n)
    radii = 1.0 + rng.normal(0, 0.05, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_two_ring(n):
    """Two-ring initialization: 10 outer + 4 inner."""
    outer_n = 10
    inner_n = n - outer_n
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _simulated_annealing_polar(params0, time_budget, rng, beta=100.0):
    """
    Simulated Annealing in polar parameter space.
    Accepts worse moves with probability exp(-delta/T).
    """
    start_time = time.time()
    params = params0.copy()
    best_params = params.copy()
    best_score = _score(_polar_to_cartesian(params))
    current_score = best_score
    
    # Temperature schedule
    T0 = 0.1
    T_end = 1e-6
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Cooldown
        t_frac = step / n_steps
        T = T0 * (T_end / T0) ** t_frac
        
        # Propose a move: perturb one or two parameters
        n_params = len(params)
        n_perturb = rng.randint(1, 4)  # Perturb 1-3 parameters
        indices = rng.choice(n_params, size=n_perturb, replace=False)
        
        new_params = params.copy()
        for idx in indices:
            # Perturbation size scales with temperature
            scale = 0.1 * T / T0
            new_params[idx] += rng.normal(0, scale)
        
        # Evaluate
        new_score = _score(_polar_to_cartesian(new_params))
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            params = new_params
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
    
    return best_params, best_score


def _lbfgs_polish_polar(params, time_budget, beta=200.0):
    """
    L-BFGS-B polish in polar parameter space using soft objective.
    """
    start_time = time.time()
    
    def obj_func(p):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        val, _ = _soft_obj_polar(p, beta)
        return val
    
    try:
        res = minimize(obj_func, params, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                              'maxfun': 2000})
        if res.x is not None:
            cand_pts = _polar_to_cartesian(res.x)
            s = _score(cand_pts)
            return res.x, s
    except Exception:
        pass
    
    return params, _score(_polar_to_cartesian(params))


def _lbfgs_polish_cartesian(pts, time_budget, beta=200.0):
    """
    L-BFGS-B polish in Cartesian space using soft objective.
    """
    start_time = time.time()
    pts_flat = pts.flatten()
    
    def obj_func(p):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        val = _soft_obj(p, beta)
        return val
    
    try:
        res = minimize(obj_func, pts_flat, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                              'maxfun': 2000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            return cand, s
    except Exception:
        pass
    
    return pts, _score(pts)


def _exact_polish(pts, time_budget):
    """
    Final polishing using exact score with Nelder-Mead.
    """
    start_time = time.time()
    pts_flat = pts.flatten()
    
    def obj_func(p):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        return _neg_score(p)
    
    try:
        res = minimize(obj_func, pts_flat, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                              'adaptive': True, 'maxfev': 4000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            return cand, s
    except Exception:
        pass
    
    return pts, _score(pts)


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_two_ring(N),
        _initial_perturbed_polygon(N, seed=1),
        _initial_perturbed_polygon(N, seed=2),
        _initial_perturbed_polygon(N, seed=3),
        _initial_perturbed_polygon(N, seed=4),
        _initial_perturbed_polygon(N, seed=5),
    ]
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    init_scores = [init_scores[i] for i in sorted_indices]
    
    best_pts = inits[0]
    best_score = init_scores[0]
    
    # Time allocation
    # Phase 1: Differential Evolution (global search) - 80s
    # Phase 2: Simulated Annealing + L-BFGS polish on top candidates - 40s
    # Phase 3: Final exact polish - 15s
    
    de_time_budget = 80.0
    sa_time_budget = 40.0
    polish_time_budget = 15.0
    
    # Phase 1: Differential Evolution in polar space
    # Parameters: r_i (14), theta_1 (1), d_thetas (13) -> 28 params
    bounds = []
    for _ in range(N):
        bounds.append((0.1, 2.0))
    bounds.append((0, 2 * np.pi))
    for _ in range(N - 1):
        bounds.append((0.1, 2 * np.pi))
    
    de_start = time.time()
    de_candidates = []
    
    try:
        stop_flag = [False]
        
        def de_callback(xk, convergence):
            if time.time() - de_start > de_time_budget:
                stop_flag[0] = True
            return stop_flag[0]
        
        res = differential_evolution(
            _neg_score_polar, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            callback=de_callback,
            polish=False
        )
        
        if res.x is not None:
            cand_pts = _polar_to_cartesian(res.x)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_pts = cand_pts
            de_candidates.append((res.x, s))
            
    except Exception:
        pass
    
    # Phase 2: Simulated Annealing + L-BFGS on top candidates
    # We'll run SA on the best DE result and a few initial configs
    sa_start = time.time()
    rng = np.random.RandomState(123)
    
    # Convert best points to polar params for SA
    def _cartesian_to_polar(pts):
        """Convert Cartesian points to polar params (r, theta_start, d_thetas)."""
        n = len(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        radii = np.sqrt(x**2 + y**2)
        thetas = np.arctan2(y, x)
        thetas = thetas % (2 * np.pi)
        
        # Sort by angle
        order = np.argsort(thetas)
        radii = radii[order]
        thetas = thetas[order]
        
        theta_start = thetas[0]
        d_thetas = np.diff(thetas)
        d_thetas = np.append(d_thetas, 2 * np.pi - thetas[-1] + thetas[0])
        
        params = np.concatenate([radii, [theta_start], d_thetas])
        return params
    
    # SA on best DE result
    if time.time() - sa_start < sa_time_budget - 5:
        try:
            best_de_params = _cartesian_to_polar(best_pts)
            sa_budget = min(sa_time_budget - 10, time.time() - sa_start)
            if sa_budget > 5:
                sa_params, sa_score = _simulated_annealing_polar(
                    best_de_params, sa_budget, rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_pts = _polar_to_cartesian(sa_params)
        except Exception:
            pass
    
    # SA on top initial configs
    for i in range(min(3, len(inits))):
        if time.time() - sa_start > sa_time_budget - 5:
            break
        try:
            init_params = _cartesian_to_polar(inits[i])
            sa_budget = min(10.0, sa_time_budget - 5 - (time.time() - sa_start))
            if sa_budget > 2:
                sa_params, sa_score = _simulated_annealing_polar(
                    init_params, sa_budget, rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_pts = _polar_to_cartesian(sa_params)
        except Exception:
            pass
    
    # L-BFGS polish on best result
    if time.time() - start_time < DEADLINE - 10:
        try:
            lbfgs_budget = min(20.0, DEADLINE - 15 - (time.time() - start_time))
            if lbfgs_budget > 3:
                best_pts, s = _lbfgs_polish_cartesian(best_pts, lbfgs_budget, beta=200.0)
                if s > best_score:
                    best_score = s
        except Exception:
            pass
    
    # Phase 3: Final exact polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            exact_budget = min(10.0, DEADLINE - 5 - (time.time() - start_time))
            if exact_budget > 2:
                best_pts, s = _exact_polish(best_pts, exact_budget)
                if s > best_score:
                    best_score = s
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
