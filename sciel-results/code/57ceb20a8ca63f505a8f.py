# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _score(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2


def _project_to_sphere(pts: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Project points onto sphere of radius R."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    return pts / norms * R


def _fibonacci_sphere(n: int, R: float = 1.0) -> np.ndarray:
    """Generate n points on a sphere using fibonacci lattice."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z**2, 0, 1))
    theta = 2 * np.pi * i / golden
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z]) * R


def _optimize_from_start(points: np.ndarray, time_limit: float) -> np.ndarray:
    """Optimize points to maximize d_min/d_max using SLSQP on sphere."""
    n = points.shape[0]
    best_points = _project_to_sphere(points.copy())
    best_score = _score(best_points)
    
    start_time = time.time()
    
    # Objective: minimize -log(d_min) + log(d_max) to maximize ratio
    # Actually, let's minimize -d_min / d_max directly, or use a smoother formulation
    # Use: minimize -(d_min / d_max) which is equivalent to maximizing the ratio
    
    def objective(flat: np.ndarray) -> float:
        pts = flat.reshape(n, 3)
        # Project to sphere to maintain constraint
        pts = _project_to_sphere(pts)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 1.0  # Penalize
        return -(d_min / d_max)
    
    # Constraints: each point on unit sphere
    constraints = []
    for i in range(n):
        constraints.append({
            'type': 'eq',
            'fun': lambda flat, i=i: np.dot(flat[3*i:3*i+3], flat[3*i:3*i+3]) - 1.0,
            'jac': lambda flat, i=i: np.array([0 if j != 3*i+k else 2*flat[j] for j in range(3*n) for k in range(3)])
        })
    
    # Try multiple random perturbations
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Perturb the starting point
        if attempt == 0:
            current = _project_to_sphere(points.copy())
        else:
            current = _project_to_sphere(points + 0.1 * np.random.randn(n, 3))
        
        flat = current.flatten()
        
        try:
            res = minimize(
                objective,
                flat,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        if res.success:
            new_points = _project_to_sphere(res.x.reshape(n, 3))
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    
    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    np.random.seed(42)
    
    # Start with fibonacci sphere
    points = _fibonacci_sphere(14, 1.0)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Optimize from fibonacci start
    if time.time() < deadline - 30:
        optimized = _optimize_from_start(points, time_limit=60)
        opt_score = _score(optimized)
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized
    
    # Try a few random starts with optimization
    for seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline - 30:
            break
        np.random.seed(seed)
        random_points = np.random.randn(14, 3)
        random_points = _project_to_sphere(random_points)
        
        # Quick local optimization
        if time.time() < deadline - 30:
            try:
                opt = _optimize_from_start(random_points, time_limit=20)
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt
            except Exception:
                pass
    
    return best_points


# EVOLVE-BLOCK-END
