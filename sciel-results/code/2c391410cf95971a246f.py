"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _min_triangle_area_vec(points):
    """
    Vectorized computation of minimum triangle area.
    points: array of shape (n, 2)
    Returns the minimum area over all C(n,3) triangles.
    """
    n = len(points)
    if n < 3:
        return float('inf')
    
    # Compute all pairwise cross products for triangles
    # For each triple (i, j, k), area = 0.5 * |cross(pj-pi, pk-pi)|
    min_a = float('inf')
    
    # Use vectorized approach for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorized computation for all k > j
            k_indices = np.arange(j + 1, n)
            if len(k_indices) == 0:
                continue
            
            pi = points[i]
            pj = points[j]
            pk = points[k_indices]
            
            # Vectorized cross product
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_a = min(min_a, np.min(areas))
    
    return min_a

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    # Constraints:
    # y >= 0
    # y <= h (max height)
    # y <= sqrt(3) * x  (left side, slope = h/0.5 = sqrt(3))
    # y <= sqrt(3) * (1 - x)  (right side)
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    # From y <= sqrt(3)*x => x >= y/sqrt(3)
    x_min_left = y / np.sqrt(3)
    # From y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area_vec(points)

def _soft_min_area(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and can be used with gradient-based optimizers.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all triangle areas
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                a = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(a)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Log-sum-exp approximation of min
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0
    
    # Use negative log-sum-exp to approximate min
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    lse = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -lse
    
    return soft_min

def _objective_smooth(x_flat, beta=50.0):
    """Objective function for gradient-based optimization (smooth approximation)."""
    pts = np.array(x_flat).reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(x_flat):
    """Objective function for exact minimum area (non-smooth)."""
    pts = np.array(x_flat).reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_triangle_area_vec(pts)

def _gradient_refine(points, max_iter=500, lr=0.01):
    """
    Perform gradient-based refinement using finite differences.
    Returns (refined_points, score).
    """
    x0 = np.array(points).flatten()
    n = len(points)
    
    def grad(x):
        """Compute gradient via finite differences."""
        dx = 1e-5
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += dx
            x_minus = x.copy()
            x_minus[i] -= dx
            g[i] = (_objective_smooth(x_plus) - _objective_smooth(x_minus)) / (2 * dx)
        return g
    
    x = x0.copy()
    best_x = x0.copy()
    best_val = _objective_hard(x0)
    
    for it in range(max_iter):
        if time.time() > _deadline - 10:
            break
        
        g = grad(x)
        x_new = x - lr * g
        x_new = np.array([_clamp_to_triangle(x_new[i*2:(i+1)*2]) for i in range(n)]).flatten()
        
        val = _objective_hard(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995  # Decay learning rate
    
    refined_points = _clamp_all(best_x.reshape(-1, 2))
    return refined_points, _evaluate(refined_points)

def _nelder_mead_refine(points, max_iter=1000):
    """
    Perform local optimization using Nelder-Mead.
    Returns (optimized_points, score).
    """
    x0 = np.array(points).flatten()
    
    def objective(x_flat):
        pts = np.array(x_flat).reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_min_triangle_area_vec(pts)
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    opt_points = np.array(result.x).reshape(-1, 2)
    opt_points = _clamp_all(opt_points)
    return opt_points, _evaluate(opt_points)

def _simulated_annealing(points, t0=0.1, t_end=0.001, n_iter=2000, cool_rate=0.999):
    """
    Perform simulated annealing optimization.
    Returns (optimized_points, score).
    """
    current_points = np.array(points).copy()
    current_score = _evaluate(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    t = t0
    n = len(current_points)
    
    for it in range(n_iter):
        if time.time() > _deadline - 10:
            break
        
        # Generate perturbation
        new_points = current_points.copy()
        for i in range(n):
            if np.random.random() < 0.3:
                new_points[i] += t * np.random.randn(2)
        
        new_points = _clamp_all(new_points)
        new_score = _evaluate(new_points)
        
        # Accept if better or with probability exp(-(delta)/t)
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            current_points = new_points
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        t *= cool_rate
    
    return best_points, best_score

def _generate_seeds():
    """Generate diverse initial point configurations."""
    h = np.sqrt(3) / 2
    
    configs = []
    
    # Configuration 1: Vertices + centroid + symmetric points
    config1 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4],
    ]
    configs.append(config1)
    
    # Configuration 2: More spread out
    config2 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.2, h/5],
        [0.8, h/5],
        [0.5, h/2],
        [0.1, h/10],
        [0.9, h/10],
        [0.35, h/4],
        [0.65, h/4],
    ]
    configs.append(config2)
    
    # Configuration 3: Random symmetric
    np.random.seed(42)
    config3 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3 + 0.03 * np.random.randn()],
        [0.25 + 0.03 * np.random.randn(), h/6 + 0.02 * np.random.randn()],
        [0.75 + 0.03 * np.random.randn(), h/6 + 0.02 * np.random.randn()],
        [0.5 + 0.03 * np.random.randn(), h/2 + 0.02 * np.random.randn()],
        [0.125 + 0.03 * np.random.randn(), h/12 + 0.02 * np.random.randn()],
        [0.875 + 0.03 * np.random.randn(), h/12 + 0.02 * np.random.randn()],
        [0.375 + 0.03 * np.random.randn(), h/4 + 0.02 * np.random.randn()],
        [0.625 + 0.03 * np.random.randn(), h/4 + 0.02 * np.random.randn()],
    ]
    configs.append(config3)
    
    # Configuration 4: Random
    np.random.seed(123)
    config4 = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config4.append([x, y])
    configs.append(config4)
    
    return configs

_deadline = None

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    global _deadline
    start_time = time.time()
    _deadline = start_time + 135  # Leave 15s safety margin
    
    h = np.sqrt(3) / 2
    
    # Generate initial configurations
    initial_configs = _generate_seeds()
    
    best_points = np.array(initial_configs[0])
    best_score = _evaluate(best_points)
    
    # Optimize each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > _deadline - 30:
            break
        
        points = _clamp_all(np.array(config))
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points
            best_score = current_score
        
        # Try multiple optimization methods
        if time.time() < _deadline - 20:
            # Nelder-Mead
            opt_points, opt_score = _nelder_mead_refine(points, max_iter=800)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
        
        if time.time() < _deadline - 15:
            # Simulated Annealing
            sa_points, sa_score = _simulated_annealing(best_points, t0=0.05, t_end=0.001, n_iter=1500, cool_rate=0.9995)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Try additional random restarts if time permits
    np.random.seed(456)
    for restart in range(15):
        if time.time() > _deadline - 25:
            break
        
        # Generate random points
        points = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = h * v
            points.append([x, y])
        
        # Also try structured random
        if restart % 2 == 0:
            base = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, h/3 + 0.05 * np.random.randn()],
                [0.25 + 0.05 * np.random.randn(), h/6 + 0.03 * np.random.randn()],
                [0.75 + 0.05 * np.random.randn(), h/6 + 0.03 * np.random.randn()],
                [0.5 + 0.05 * np.random.randn(), h/2 + 0.03 * np.random.randn()],
                [0.125 + 0.05 * np.random.randn(), h/12 + 0.03 * np.random.randn()],
                [0.875 + 0.05 * np.random.randn(), h/12 + 0.03 * np.random.randn()],
                [0.375 + 0.05 * np.random.randn(), h/4 + 0.03 * np.random.randn()],
                [0.625 + 0.05 * np.random.randn(), h/4 + 0.03 * np.random.randn()],
            ]
            points = base
        
        points = _clamp_all(np.array(points))
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points
            best_score = current_score
        
        if time.time() < _deadline - 15:
            # Nelder-Mead
            opt_points, opt_score = _nelder_mead_refine(points, max_iter=600)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
        
        if time.time() < _deadline - 10:
            # Simulated Annealing
            sa_points, sa_score = _simulated_annealing(best_points, t0=0.03, t_end=0.001, n_iter=1000, cool_rate=0.9995)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Final gradient refinement if time permits
    if time.time() < _deadline - 5:
        grad_points, grad_score = _gradient_refine(best_points, max_iter=200, lr=0.005)
        if grad_score > best_score:
            best_points = grad_points
            best_score = grad_score
    
    # Final check: ensure all points are inside the triangle
    best_points = _clamp_all(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
