"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

N = 13

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(0.5 * cross)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area

def _generate_points(params):
    """
    Generate 13 points from parameters.
    params: [A1, phi1, A2, phi2]
    Points are on a unit circle with angular perturbations:
    theta_k = 2*pi*k/N + A1*cos(2*pi*k/N + phi1) + A2*cos(4*pi*k/N + phi2)
    """
    n = 13
    A1, phi1, A2, phi2 = params
    
    base_angles = np.arange(n) * 2 * np.pi / n
    perturb = A1 * np.cos(base_angles + phi1) + A2 * np.cos(2 * base_angles + phi2)
    angles = base_angles + perturb
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _cma_es(objective, x0, sigma, max_iter=100, pop_size=None):
    """
    Simple CMA-ES implementation.
    objective: function to minimize.
    x0: initial mean.
    sigma: initial step size.
    """
    n = len(x0)
    if pop_size is None:
        pop_size = int(4 + 3 * np.log(n))
    
    mean = np.array(x0, dtype=float)
    C = np.eye(n)
    C_sqrt = np.eye(n)
    step_size = sigma
    pc = np.zeros(n)
    cs = 1.0 / (pop_size + n + 2)
    cmu = 2 / ((n + 1.3) ** 2 + pop_size)
    c1 = 2 / ((n + 1) ** 2 + 10)
    cmu = min(cmu, 2 * (1 - c1) / ((n + 2) ** 2 - 1))
    cc = (4 + n / 4) / (n + 4 + 2 * pop_size / n)
    c_sigma = (5 + n / 2) / (n + 5 + pop_size)
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    best_x = mean.copy()
    best_f = objective(mean)
    
    for iteration in range(max_iter):
        # Generate population
        pop = []
        for i in range(pop_size):
            z = np.random.randn(n)
            x = mean + step_size * (C_sqrt @ z)
            pop.append(x)
        
        # Evaluate
        fvals = np.array([objective(x) for x in pop])
        idx = np.argsort(fvals)
        pop = np.array([pop[i] for i in idx])
        fvals = fvals[idx]
        
        # Update mean
        w = np.log(pop_size / 2) - np.arange(pop_size)
        w /= w.sum()
        lambda_eff = 1.0 / np.sum(w[:int(0.5 * pop_size)] ** 2)
        mu_eff = min(pop_size, lambda_eff)
        w = w[:mu_eff]
        w = w / w.sum()
        
        new_mean = np.sum(w[:, None] * pop[:mu_eff], axis=0)
        
        # Update covariance matrix
        pc = (1 - cc) * pc + cc * (new_mean - mean) / step_size
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * cc) * (new_mean - mean) / step_size
        
        rank_one = np.outer(pc, pc)
        rank_mu = np.zeros((n, n))
        for i in range(mu_eff):
            y = (pop[i] - mean) / step_size
            rank_mu += w[i] * np.outer(y, y)
        
        C = (1 + c1 * (rank_one - C) + cmu * (rank_mu - C))
        
        # Ensure C is positive definite
        C = (C + C.T) / 2
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Fallback if not PD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = np.linalg.cholesky(C)
        
        # Update step size
        step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
        
        mean = new_mean
        
        # Track best
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = pop[0].copy()
            
        # Convergence check
        if step_size < 1e-12:
            break
            
    return best_x, -best_f

def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses CMA-ES to optimize angular perturbations.
    """
    best_score = 0.0
    best_points = None
    
    # Parameters: [A1, phi1, A2, phi2]
    # A1, A2 are amplitudes of angular perturbation (radians)
    # phi1, phi2 are phases
    
    rng = np.random.RandomState(42)
    
    # Initialize with regular polygon (A1=A2=0)
    init_params = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Run CMA-ES
    # Objective function for CMA-ES (minimizes negative score)
    def obj(params):
        points = _generate_points(params)
        return -_score(points)
    
    try:
        best_params, best_neg_score = _cma_es(obj, init_params, sigma=0.1, max_iter=150)
        points = _generate_points(best_params)
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points
    except Exception as e:
        pass
    
    # If CMA-ES fails or doesn't improve, try a few random restarts with Nelder-Mead or just return best
    # Since directive asks for CMA-ES, we rely on that. 
    # But to be safe, if score is 0, fallback to regular polygon.
    if best_points is None or best_score == 0.0:
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
