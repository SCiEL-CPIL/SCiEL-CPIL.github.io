"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _compute_min_triangle_area(points):
    """Vectorized computation of minimum triangle area over all triples."""
    n = len(points)
    # For each pair (i,j), compute cross product with all k > j
    # area = 0.5 * |(p_j - p_i) x (p_k - p_i)|
    min_area = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = points[j] - points[i]
            # Compute cross products with all remaining points
            v2 = points[j + 1:] - points[i]
            crosses = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
            min_area = min(min_area, 0.5 * np.min(crosses))
    return min_area


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = _compute_min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area


def _generate_points_from_angular_perturbation(params):
    """
    Generate 13 points on a unit circle with angular perturbations.
    
    params: [a1, phi1, a2, phi2]
    - a1, phi1: amplitude and phase of first harmonic angular perturbation
    - a2, phi2: amplitude and phase of second harmonic angular perturbation
    
    Angular perturbation: delta_theta(k) = a1*cos(k*2*pi/13 + phi1) + a2*cos(2*k*2*pi/13 + phi2)
    Points are placed at angle = k*2*pi/13 + delta_theta(k) on unit circle.
    
    This preserves the property that all points are on a convex curve (near-circle),
    and reduces the search space to 4 parameters.
    """
    n = 13
    a1, phi1, a2, phi2 = params[0], params[1], params[2], params[3]
    
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbation
    delta_theta = a1 * np.cos(base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + delta_theta
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def _cma_es(objective, x0, sigma, bounds, max_iter=200, pop_size=50, seed=42):
    """
    CMA-ES optimizer implementation.
    
    objective: function to minimize
    x0: initial mean
    sigma: initial step size
    bounds: list of (low, high) for each parameter
    max_iter: maximum number of generations
    pop_size: population size
    seed: random seed
    """
    rng = np.random.RandomState(seed)
    n = len(x0)
    
    mean = np.array(x0, dtype=float)
    sigma = float(sigma)
    
    # Covariance matrix (start as identity)
    C = np.eye(n)
    # Eigenvalue decomposition for square root
    eigvals, eigvecs = np.linalg.eigh(C)
    C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
    
    # Strategy parameters
    lambda_ = pop_size
    mu = lambda_ // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
    chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
    
    # Evolution paths
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    
    best_x = mean.copy()
    best_f = objective(mean)
    
    for it in range(max_iter):
        # Generate population
        pop = np.zeros((lambda_, n))
        for i in range(lambda_):
            z = rng.standard_normal(n)
            pop[i] = mean + sigma * (C_sqrt @ z)
        
        # Clip to bounds
        for i in range(lambda_):
            for j in range(n):
                pop[i, j] = np.clip(pop[i, j], bounds[j][0], bounds[j][1])
        
        # Evaluate
        fitness = np.array([objective(pop[i]) for i in range(lambda_)])
        
        # Sort by fitness
        order = np.argsort(fitness)
        best_idx = order[0]
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_x = pop[best_idx].copy()
        
        # Update mean
        y_w = np.zeros(n)
        for i in range(mu):
            y_w += weights[i] * (pop[order[i]] - mean) / sigma
        
        # Update evolution path for sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ y_w)
        
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = np.clip(sigma, 1e-10, 10.0)
        
        # Update evolution path for C
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        
        # Update covariance matrix
        v = p_sigma / np.linalg.norm(p_sigma) if np.linalg.norm(p_sigma) > 1e-10 else np.zeros(n)
        rank1 = (1 + c_sigma * (d_sigma - 1) + (np.linalg.norm(p_sigma) ** 2 / chi_n ** 2 - 1) * c_sigma) * np.outer(v, v)
        
        # Rank-mu update
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            y_i = (pop[order[i]] - mean) / sigma
            rank_mu += weights[i] * np.outer(y_i, y_i)
        
        C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_c * np.outer(p_c, p_c) + c_sigma * (rank1 - C) + (2 * c_sigma / d_sigma) * rank_mu
        
        # Ensure symmetry and positive definiteness
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Update mean
        mean = mean + sigma * y_w
        
        # Clip mean to bounds
        for j in range(n):
            mean[j] = np.clip(mean[j], bounds[j][0], bounds[j][1])
    
    return best_x, best_f


def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    
    Uses angular perturbation model with 2 Fourier harmonics (4 parameters)
    optimized via CMA-ES.
    """
    best_score = 0.0
    best_points = None
    
    # Parameter bounds: [a1, phi1, a2, phi2]
    # a1, a2: angular perturbation amplitudes (radians)
    # phi1, phi2: phases
    bounds = [
        (-0.3, 0.3),      # a1
        (0, 2 * np.pi),   # phi1
        (-0.2, 0.2),      # a2
        (0, 2 * np.pi),   # phi2
    ]
    
    # Initial guess: regular polygon (no perturbation)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Run CMA-ES
    try:
        best_params, best_neg_score = _cma_es(
            objective=lambda p: -_score(_generate_points_from_angular_perturbation(p)),
            x0=x0,
            sigma=0.1,
            bounds=bounds,
            max_iter=300,
            pop_size=40,
            seed=42
        )
        
        # Clip to bounds
        for i in range(len(best_params)):
            best_params[i] = np.clip(best_params[i], bounds[i][0], bounds[i][1])
        
        points = _generate_points_from_angular_perturbation(best_params)
        
        # Verify convexity
        try:
            hull = ConvexHull(points)
            if len(hull.vertices) < 13:
                # Not convex, try with smaller perturbations
                best_params = best_params * 0.5
                points = _generate_points_from_angular_perturbation(best_params)
                hull = ConvexHull(points)
                if len(hull.vertices) < 13:
                    raise Exception("Not convex")
        except Exception:
            # Fallback to regular polygon
            angles = np.arange(N) * 2 * np.pi / N
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
    except Exception:
        # Fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    if best_points is None:
        # Final fallback: regular 13-gon
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
