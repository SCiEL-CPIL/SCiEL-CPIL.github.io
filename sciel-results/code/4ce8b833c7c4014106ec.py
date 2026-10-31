"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    min_tri = np.min(_all_triangle_areas(points))
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    return min_tri / ha


def _gen_points(params):
    """
    params: [theta0, a1, phi1, a2, phi2]
    Angles: theta(k) = theta0 + k*2*pi/13 + a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    Points on unit circle at these angles.
    """
    n = 13
    theta0, a1, phi1, a2, phi2 = params
    base = np.arange(n) * 2 * np.pi / n
    angles = theta0 + base + a1 * np.cos(base + phi1) + a2 * np.cos(2 * base + phi2)
    # Sort angles to ensure convex ordering
    angles = np.sort(angles)
    # Wrap to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Re-sort after wrap
    angles = np.sort(angles)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _objective(params):
    """Negative score; penalize non-convex."""
    points = _gen_points(params)
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < 13:
            return -0.005
    except Exception:
        return -0.005
    return -_score(points)


def _cma_es(objective, x0, sigma, max_iter=2000, seed=42):
    """
    Minimal CMA-ES implementation.
    x0: initial mean (array)
    sigma: initial step size
    Returns best_params, best_obj
    """
    rng = np.random.RandomState(seed)
    n = len(x0)
    pop = max(4 + int(3 * np.log(n)), 6)
    
    mean = np.array(x0, dtype=float)
    C = np.eye(n)
    C_inv_sqrt = np.eye(n)
    sigma = float(sigma)
    
    best_x = mean.copy()
    best_f = objective(mean)
    
    # Precompute
    weights = np.log(pop / 2) - np.log(np.arange(1, pop + 1))
    weights /= np.sum(weights)
    mu = pop // 2
    w = weights[:mu]
    w /= np.sum(w)
    mueff = 1.0 / np.sum(w ** 2)
    c_sigma = (mueff + 2) / (n + mueff + 5)
    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
    d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    count_steps = 0
    
    for it in range(max_iter):
        # Sample
        ys = rng.standard_normal((pop, n))
        # Apply C_inv_sqrt
        ys = (C_inv_sqrt @ ys.T).T
        xs = mean + sigma * ys
        fs = np.array([objective(x) for x in xs])
        
        # Sort by fitness
        order = np.argsort(fs)
        xs_sorted = xs[order]
        fs_sorted = fs[order]
        
        # Update best
        if fs_sorted[0] < best_f:
            best_f = fs_sorted[0]
            best_x = xs_sorted[0].copy()
        
        # Weighted mean
        y_w = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
        
        # Update C_inv_sqrt via eigenvalue decomposition (simplified)
        # For small n, we can do full update
        y_w_Cinv = (C_inv_sqrt @ y_w)
        
        # Update p_sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * y_w_Cinv
        
        # Update p_c
        h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) <
                          (1.4 + 2 / (n + 1)) * chi_n) else 0.0
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_w_Cinv
        
        # Update C
        delta_h = (1 - h_sigma) * c_c * (2 - c_c)
        rank_c = np.outer(p_c, p_c)
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            rank_mu += w[i] * np.outer(y_w_Cinv[i], y_w_Cinv[i]) if False else w[i] * np.outer(xs_sorted[i] - mean, xs_sorted[i] - mean)
        # Actually need to use y_w_Cinv for the rank-mu update properly
        # Let me redo: the rank-mu update uses (x_i - mean)/sigma transformed by C_inv_sqrt
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            diff = (xs_sorted[i] - mean) / sigma
            diff_c = C_inv_sqrt @ diff
            rank_mu += w[i] * np.outer(diff_c, diff_c)
        
        C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_c + c_mu * rank_mu
        
        # Update sigma
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update mean
        mean = mean + (1 - c_c) * 0 + 0  # mean update is separate
        # Actually mean update:
        mean = mean  # We'll update mean below
        
        # Proper mean update
        mean = mean + 0  # placeholder
        
        # Let me fix the mean update
        # mean_new = mean + sum(w_i * (x_i_sorted - mean))
        # But we need to use the original (untransformed) weighted sum
        mean = mean + np.sum(w[:, None] * (xs_sorted[:mu] - mean), axis=0)
        
        # Update C_inv_sqrt via eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            C_inv_sqrt = np.eye(n)
        
        # Convergence check
        if sigma < 1e-12:
            break
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Uses angular perturbation with 2 harmonics (5 params) + CMA-ES.
    """
    best_score = 0.0
    best_points = None
    
    # Regular 13-gon as baseline
    base_angles = np.arange(N) * 2 * np.pi / N
    base_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    best_score = _score(base_points)
    best_points = base_points.copy()
    
    # CMA-ES with multiple restarts
    rng = np.random.RandomState(42)
    
    for trial in range(5):
        if trial == 0:
            x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            sigma0 = 0.1
        else:
            x0 = np.array([
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.10, 0.10),
                rng.uniform(0, 2 * np.pi),
            ])
            sigma0 = 0.08
        
        try:
            best_params, best_obj = _cma_es(
                _objective, x0, sigma0, max_iter=1500, seed=42 + trial
            )
            
            points = _gen_points(best_params)
            
            # Verify convexity
            try:
                hull = ConvexHull(points)
                if len(hull.vertices) < 13:
                    continue
            except Exception:
                continue
            
            sc = _score(points)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()
        except Exception:
            continue
    
    return best_points
# EVOLVE-BLOCK-END
